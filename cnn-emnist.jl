ENV["JULIA_CUDA_SILENT"] = true
using LinearAlgebra, Statistics, Flux, MLDatasets, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf

x_treino, y_treino = MLDatasets.EMNIST(:letters,Tx=Float32, split=:train)[:] # Pega o conjunto de treinamento do dataset
x_treino          = permutedims(x_treino,(2,1,3)); # troca os eixos weight e height de lugar
x_treino          = reshape(x_treino,(28,28,1,124800)); # transforma a matriz criado em 60000 matrizes 28x28 com 1 dimensao
y_treino          = Flux.onehotbatch(y_treino, 0:36) # codifica os campos em uma matriz de 1's e 0's usando 10 possiveis classes de valores.
dados_treino       = Flux.Data.DataLoader((x_treino, y_treino), batchsize=128) # Usado para carregar os dados em lotes

x_teste, y_teste  = MLDatasets.EMNIST(:letters,Tx=Float32, split=:test)[:] # Pega o conjunto de teste do dataset
x_teste           = permutedims(x_teste,(2,1,3)); # Permuta o eixo 1 e 2
x_teste           = reshape(x_teste,(28,28,1,20800)); # Transforma os arrays criado em 10000 matrizes de 28x28
y_teste           = Flux.onehotbatch(y_teste, 0:36) # codifica os campos em uma matriz de 1's e 0's usando 10 possiveis classes de valores.

modelo = Chain(
       # 28x28 => 14x14
       Conv((5, 5), 1=>8,   pad=2, stride=2, relu), # recebera uma imagem 28x28 e fara uma convolucao com 8 filtros 5x5, e depois uma linearizacao usando relu. Com isso temos: (28 + 2*2 - 5)/2 + 1 = 14,5 => floor(14,5) = 14
       # 14x14 => 7x7 (14 + 2*1 - 3)/2 + 1 = 7,5 => floor(7,5) = 7
       Conv((3, 3), 8=>16,  pad=1, stride=2, relu),
       # 7x7 => 4x4 (7 + 2*1 - 3)/2 + 1 = 4 => floor(4) = 4
       Conv((3, 3), 16=>32, pad=1, stride=2, relu),
       # 4x4 => 2x2 (4 + 2*1 - 3)/2 + 1 = 2,5 => floor(2,5) = 2
       Conv((3, 3), 32=>32, pad=1, stride=2, relu),
       # (2 + 2*1 - 3)/2 + 1 = 1,5 
       # Average pooling on each width x height feature map
       GlobalMeanPool(), # faz o pooling usando a estratégia do global
       Flux.flatten, # faz uma linearizacao
       Dense(32, 37), # Cria uma rede totalmente conectada de entrada 32 e saida 10
       Flux.softmax ) # Por fim aplica softmax para normalizar os dados de saida e ter a probabilidade de ser cada um dos valores

acuracia(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
perda(x, y)     = Flux.crossentropy(modelo(x), y)

opt = Flux.ADAM()
ps  = Flux.params(modelo)

num_épocas = 11

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
end

ŷtreino =   modelo(x_treino)
ŷteste  =   modelo(x_teste)

acuracia(ŷtreino, y_treino)
acuracia(ŷteste, y_teste)

Flux.onecold(y_treino[:,2]) - 1  # rótulo da amostra 2
plot(Gray.(x_treino[:,:,1,2]))   # imagem da amostra 2

cm = ConfusionMatrix()
fit!(cm, Flux.onecold(y_teste) .-1, Flux.onecold(ŷteste) .-1)
print(cm)

res = info(cm)

heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white,:blue]),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")

# Limita o mapa de cores, para vermos melhor onde os erros estão

heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white,:blue]),
        clim=(0., 0.02),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")





