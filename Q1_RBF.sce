// TERCEIRO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 1
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
//-----------------------------------------------------------------------------

clear;
clc;
clf;
base = fscanfMat('twomoons.dat');

//PARTE 1: Cálculo das centróides usando K-means clustering

//Separa 90% dos dados para treino e 10% dos dados para teste
//Base de treino - 450 amostras da primeira metade da base e 450 da última
base_treino = base(1:450,:);
base_treino(451:900,:) = base(500:949,:);
//Base de teste - 50 amostras da primeira metade da base e 51 da última
base_teste = base(451:499,:);
base_teste(50:101,:) = base(950:1001,:);

//Define as entradas da rede (X) e a saída esperada (D)
X = base_treino(:,1:2);    //Entradas da rede
D = base_treino(:,3);      //Saídas da rede

//Implementa k-means
//cria duas centroides aleatorias - primeira coluna eixo X e segunda coluna eixo Y
//cada linha uma centroide
centroides = rand(2, 2) .* (max(X) - min(X)) + min(X);

//Calcula a distancia de todos os pontos para a centroide 1
distancias(:,1) = sqrt(((centroides(1,1)-X(:,1))^2) + ((centroides(1,2)-X(:,2))^2));
//Calcula a distancia de todos os pontos para a centroide 2
distancias(:,2) = sqrt(((centroides(2,1)-X(:,1))^2) + ((centroides(2,2)-X(:,2))^2));
//Testa qual a menor distância e classifica o ponto como cluster 1 ou 2.
X_classif = X; //copia a matriz de entrada para uma nova que terá a classificação
for i=1:900
    [min_valor, min_linha] = min(distancias(i,:));
    X_classif(i,3) = min_linha;
end
//Entra em loop de classificação até que as centroides não se movam mais

//calcula novas posições para as centroides
somax1=0;
somay1=0;
somax2=0;
somay2=0;

centroides_atuais = centroides;
centroides_anteriores = [0,0,0,0];
centroides_temporarias = [0,0,0,0];

while centroides_atuais <> centroides_anteriores,
//clf;
for i=1:900

    if X_classif(i,3) == 1 then
        somax1 = (somax1+X_classif(i,1));
        somay1 = (somay1+X_classif(i,2));
    elseif X_classif(i,3) == 2 then
        somax2 = (somax2+X_classif(i,1));
        somay2 = (somay2+X_classif(i,2));
    end
end
mediax1 = somax1/900;
mediay1 = somay1/900;
mediax2 = somax2/900;
mediay2 = somay2/900;

somax1 = 0;
somay1 = 0;
somax2 = 0;
somay2 = 0;

centroides_temporarias = [mediax1, mediay1; mediax2, mediay2];
centroides_anteriores = centroides_atuais;
centroides_atuais = centroides_temporarias;

end

//Cria duas matrizes classificadas e plota o gráfico classificado pelo k-means
k=1;
j=1;

for i=1:900

    if X_classif(i,3) == 1 then
        X_1(k,:) = X_classif(i,:);
        k = k+1
    elseif X_classif(i,3) == 2 then
        X_2(j,:) = X_classif(i,:);
        j = j+1
    end
end

cluster1x = X_1(:,1);
cluster1y = X_1(:,2);
scatter(cluster1x,cluster1y,26,"scilabred3","fill", ".");

cluster2x = X_2(:,1);
cluster2y = X_2(:,2);
scatter(cluster2x,cluster2y,26,"scilabgreen3","fill", ".");
xtitle("Gráfico twomoons.dat clusterizado pelo k-means. Cada cor representa um cluster")

//FIM DA PARTE 1

//PARTE 2: Implementação da rede RBF

//Par de RBF de saída do neurônio
for i=1:900
    G(i,1) = 1;
    if X_classif(i,3) == 1 then
        G(i,2) = exp(-(sqrt(((X_classif(i,1)-centroides(1,1))^2) + ((X_classif(i,2)-centroides(1,2))^2)))^2);
    elseif X_classif(i,3) == 2 then
        G(i,3) = exp(-(sqrt(((X_classif(i,1)-centroides(2,1))^2) + ((X_classif(i,2)-centroides(2,2))^2)))^2);
    end
end

// Matriz de pesos W

W = [(((G' * G) \ G') * D(:, 1))'];

// Calcula a saída da rede (d)

d = G * W';

//FIM DA PARTE 2

//PARTE 3: Mostra as saídas da rede no console

disp("---------------------------- REDE RBF  ----------------------------");
disp("-------- 2 neurônios na camada oculta + bias e 1 de saída ---------")
disp("Matriz de centroides encontrada (centroides_atuais) (onde cada linha é uma centróide)");
disp(centroides_atuais);
disp("Obs.: Método usado para achar as centróides: K-means clustering");
disp("----------------------------------------------")
disp("Matriz de pesos (W) encontrada (onde cada elemento é um peso)");
disp(W);
disp("----------------------------------------------")
disp("Matriz de saídas das duas funções: G");
disp("Obs.: Cada coluna é a saída de um neurônio e a primeira é o bias");
disp("----------------------------------------------")
disp("Matriz de saídas da rede: d");
disp("Obs.: Cada linha representa a saída de um dado input");
disp("----------------------------------------------")

//FIM DA PARTE 3
