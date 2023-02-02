ENV["JULIA_CUDA_SILENT"] = true
using LinearAlgebra, Statistics, Flux, MLDatasets, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf
using Zygote
using BSON

function load_train_dataset()
	# Pega o conjunto de treinamento do dataset
    x_treino, y_treino = MLDatasets.EMNIST(:letters, Tx=Float32, split=:train)[:] 
	# troca os eixos weight e height de lugar
    x_treino = permutedims(x_treino, (2, 1, 3)) 
	# transforma a matriz criada em 124800 matrizes 28x28 com 1 dimensao
    x_treino = reshape(x_treino, (28, 28, 1, 124800)) 
	# codifica os campos em uma matriz de 1's e 0's usando 37 possiveis classes de valores.
    y_treino = Flux.onehotbatch(y_treino, 0:36) 
    return x_treino, y_treino
end

function load_test_dataset()
	# Pega o conjunto de teste do dataset
    x_teste, y_teste = MLDatasets.EMNIST(:letters, Tx=Float32, split=:test)[:] 
	# Permuta os eixos weight e height de lugar
    x_teste = permutedims(x_teste, (2, 1, 3)) 
	# Transforma os arrays criado em 20800 matrizes de 28x28
    x_teste = reshape(x_teste, (28, 28, 1, 20800)) 
	# codifica os campos em uma matriz de 1's e 0's usando 37 possiveis classes de valores.
    y_teste = Flux.onehotbatch(y_teste, 0:36) 
    return x_teste, y_teste
end

function create_model()
    return Chain(
		# Recebera uma imagem 28x28 e fara uma convolucao com 16 filtros 3x3, 
		# e depois uma linearizacao usando relu. 
		# Com isso temos: (28 + 2*1 - 3)/1 + 1 = 28 => floor(28) = 28
        Conv((3, 3), 1 => 16, pad=(1, 1), relu), 
		# Recebera 26 imagems 28x28 e fara um pooling 2,2 para diminuirmos o tamanho
		# da imagem e melhorar o processamento, o que resultará em uma imagem 14x14.
        MaxPool((2, 2)),
		# Recebera 16 imagens 14x14 e fara uma convolucao com 32 filtros 3x3, 
		# e depois uma linearizacao usando relu. 
		# Com isso temos: (14 + 2*1 - 3)/1 + 1 = 14 => floor(14) = 14
        Conv((3, 3), 16 => 32, pad=(1, 1), relu),
		# Recebera 32 imagems 14x14 e fara um pooling 2,2 para diminuirmos o tamanho
		# da imagem e melhorar o processamento, o que resultará em uma imagem 7x7.
        MaxPool((2, 2)),
		# Recebera 32 imagens 7x7 e fara uma convolucao com 32 filtros 3x3, 
		# e depois uma linearizacao usando relu.
		# Com isso temos: (7 + 2*1 - 3)/1 + 1 = 7 => floor(7) = 7
        Conv((3, 3), 32 => 32, pad=(1, 1), relu),
		# Recebera 32 imagems 7x7 e fara um pooling 2,2 para diminuirmos o tamanho
		# da imagem e melhorar o processamento, o que resultará em uma imagem 3x3.
        MaxPool((2, 2)),
		# Faz uma linearizacao dos dados
        Flux.flatten, 
		# Cria uma rede totalmente conectada de entrada 288 e saida 37 que é o número de classes
        Dense(288, 37), 
		# Aplica sotfmax para dar a probabilidade de ser cada uma das 37 classes
        Flux.softmax, 
    )
end

function print_confusion_matrix(x_treino, x_teste, y_treino, y_teste)
    ŷtreino = modelo(x_treino)
    ŷteste = modelo(x_teste)

    acuracia(ŷtreino, y_treino)
    acuracia(ŷteste, y_teste)

    Flux.onecold(y_treino[:, 2]) - 1  # rótulo da amostra 2
    plot(Gray.(x_treino[:, :, 1, 2]))   # imagem da amostra 2

    cm = ConfusionMatrix()
    fit!(cm, Flux.onecold(y_teste) .- 1, Flux.onecold(ŷteste) .- 1)
    print(cm)

    res = info(cm)

    heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white, :blue]),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")

    # Limita o mapa de cores, para vermos melhor onde os erros estão

    heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white, :blue]),
        clim=(0.0, 0.02),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")
end

acuracia(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))

function train()
    modelo = create_model()

    perda(x, y) = Flux.crossentropy(modelo(x), y)

    opt = Flux.ADAM()
    ps = Flux.params(modelo)

    num_épocas = 8

    x_treino, y_treino = load_train_dataset()
    x_teste, y_teste = load_test_dataset()

	dados_treino = Flux.Data.DataLoader((x_treino, y_treino), batchsize=128)

    @info("Starting the training...")
    best_acc = 0.0
    last_improvement = 0
    for época in 1:num_épocas
        println("Época ", época)
        Flux.train!(perda, ps, dados_treino, opt)

        # Calcule a acurácia:
        ŷteste = modelo(x_teste)
        acu = acuracia(ŷteste, y_teste)

        @info(@sprintf("[%d]: Acurácia nos testes: %.4f", época, acu))
        # Se a acurácia for muito boa, termine o treino
        if acu >= 0.999
            @info(" -> Término prematuro: alcançamos uma acurácia de 99.9%")
            break
        end

        if acu >= best_acc
            @info("We have a new best accuracy! Saving it...")
            BSON.@save "emnist_conv.bson" modelo
            best_acc = acu
            last_improvement = época
        end

    end
end

# Testing the model, from saved model
function test()

    @info "Loading the test data"
    x_teste, y_teste = load_test_dataset()

    @info "Re-constructing the model with random initial weights"
    newModel = create_model()

    @info "Loading saved model"
    BSON.@load "emnist_conv.bson" modelo

    @info "Loading parameters onto the model"
    Flux.loadmodel!(newModel, modelo)

    model_test = newModel(x_teste)
    @info "Getting the accuracy"
    acu = acuracia(model_test, y_teste)
	@info(@sprintf("Acurácia nos testes: %.4f", acu))
end

