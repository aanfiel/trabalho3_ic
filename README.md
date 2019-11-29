# 28/11/2019: Trabalho 3 de inteligência computacional
### Este repositório hospeda os arquivos necessários à execução do trabalho 3 realizado durante os estudos da disciplina de inteligência computacional

# Universidade Federal do Ceará
### Inteligência Computacional
### Professor: Jarbas Joaci de Mesquita Sá Júnior

### Construído em:

* [Scilab 6.0.2](https://www.scilab.org/) - Software open source para computação numérica
* [Linux Mint 19.2 Tina](https://www.linuxmint.com/) - Sistema operacional usado

### Autor:

* **José Lopes de S. Filho** - [LinkedIn](https://www.linkedin.com/in/joselopesfilho/)
* *Engenharia da Computação (UFC) - Matrícula # 389097*

### Licença:

Este projeto é licenciado sob a MIT License - ver o arquivo [LICENSE](LICENSE) para detalhes

## Questão 1

*1.  Implemente uma rede neural RBF para traçar uma superfície de decisão nas amostras do conjunto de dados twomoons.dat.*

*Obs.: na primeira coluna da base de dados constam as medidas da variável x1 e na segunda coluna as medidas da variável x2. A classe de cada vetor de medidas (x1, x2) é dada na terceira coluna.*

### Iniciando 
Para a resolução desta questão e criação deste relatório foram usados os seguintes arquivos:

* [Q1_RBF.sce](Q1_RBF.sce) - Código-fonte da aplicação proposta na questão
* [twomoons.dat](twomoons.dat) - Conjunto de dados da questão
* [Q1_RBF_img1.png](Q1_RBF_img1.png) - Gráfico dos dados de treinamento twomoons clusterizado pelo k-means para encontrar as centroides
* [Q1_RBF_img2.png](Q1_RBF_img2.png) - Retorno do console ao executar o código

## Código comentado

```
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
```
#PARTE 1: Cálculo das centróides usando K-means clustering
```
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
```
#PARTE 2: Implementação da rede RBF
```
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
```
#PARTE 3: Mostra as saídas da rede no console
```
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
```

## Discussão dos resultados obtidos

Ao executar o arquivo [tr2_q1_elm.sce](tr2_q1_elm.sce) no Scilab, podemos verificar basicamente duas ações: 
* A abertura da janela gráfica exibindo o gráfico da função da questão:

![grafico_7n](https://user-images.githubusercontent.com/51038132/68537069-df730680-033c-11ea-8a5a-449c099c77fe.png)

* O console irá retornar o valor do grau de adaptação aos dados:

![console_t2_q1](https://user-images.githubusercontent.com/51038132/68537102-85bf0c00-033d-11ea-92be-b8b010801cec.png)

* Ao alterar o número de neurônios, o grau de adaptação aos dados se altera bastante, atingindo um ponto ótimo com 7 neurônios. Encontrar o número ideal de neurônios da camada escondida não é uma tarefa fácil porque depende de uma série de fatores, muito dos quais não temos controle total. O valor de *q* é geralmente encontrado por tentativa-e-erro, em função da capacidade de generalização da rede.

![grafico_1n](https://user-images.githubusercontent.com/51038132/68537068-df730680-033c-11ea-94e8-a79bacf27eb9.png)

Retorno do gráfico com 1 neurônio usado.
![grafico_7n](https://user-images.githubusercontent.com/51038132/68537069-df730680-033c-11ea-8a5a-449c099c77fe.png)

Retorno do gráfico com 7 neurônios usados.
![grafico_17n](https://user-images.githubusercontent.com/51038132/68537070-df730680-033c-11ea-9026-b8e4533f8d3e.png)

Retorno do gráfico com 17 neurônios usados.

## Questão 2

*2. Classifique o conjunto de dados disponível no arquivo iris_log.dat usando: K-NN, Nearest Prototype Classifier, Perceptron, MLP, e ELM. Utilize normalização zscore e a estratégia de validação cruzada leave-one-out.* 

*Obs.: 1. É permitido usar funções já disponíveis para o MLP. Os demais classificadores deverão ser codificados. 
2. Na base iris_log.dat, as quatro primeiras colunas representam os atributos dos vetores de características e as três últimas representam a classe da amostra ([1 0 0], [0 1 0] e [0 0 1]).*

### Parte 1: K-NN
### Iniciando 
Para a resolução desta questão e criação deste relatório foram usados os seguintes arquivos:

* [tr2_q2_knn.sce](tr2_q2_knn.sce) - Código-fonte da classificação usando K-NN (K-Nearest Neighbors)
* [iris_log.dat](iris_log.dat) - Conjunto de dados da questão
* [tr2_q2_knn_1.png](tr2_q2_knn_1.png) - Imagem 1 do console para uso do K-NN
* [tr2_q2_knn_2.png](tr2_q2_knn_2.png) - Imagem 2 do console para uso do K-NN
* [tr2_q2_knn_3.png](tr2_q2_knn_3.png) - Imagem 3 do console para uso do K-NN
* [tr2_q2_knn_4.png](tr2_q2_knn_4.png) - Imagem 4 do console para uso do K-NN
* [tr2_q2_knn_5.png](tr2_q2_knn_5.png) - Imagem 5 do console para uso do K-NN
* [tr2_q2_knn_6.png](tr2_q2_knn_6.png) - Imagem 6 do console para uso do K-NN

## Código comentado

```
// SEGUNDO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 2 (KNN)
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
//-----------------------------------------------------------------------------

//PARTE 1: AJUSTES INICIAIS

clc;    //Limpa a tela
clear;  //Limpa as variáveis armazenadas anteriormente

//Importando a base de dados iris_log.dat para a variável ibase2
ibase2 = fscanfMat('iris_log.dat');

/*separando a primeira entrada para teste posterior (leave-one-out)
//para teste posterior basta entrar com os valores (5.1, 3.5, 1.4, 2.0) 
e verificar retorno 1 0 0 (flor setosa) nas entradas quando solicitado*/
ibase3 = ibase2(2:150,:);

//Normalização z-score da base de dados
C1 = mean(ibase3(:,1));     //Média da primeira coluna dos dados
P1 = stdev(ibase3(:,1));    //Desvio padrão da primeira coluna dos dados
C2 = mean(ibase3(:,2));     //Média da segunda coluna dos dados
P2 = stdev(ibase3(:,2));    //Desvio padrão da segunda coluna dos dados
C3 = mean(ibase3(:,3));     //Média da terceira coluna dos dados
P3 = stdev(ibase3(:,3));    //desvio padrão da terceira coluna dos dados
C4 = mean(ibase3(:,4));     //Média da quarta coluna dos dados
P4 = stdev(ibase3(:,4));    //Desvio padrão da quarta coluna dos dados
ibase(:,1) = (ibase3(:,1)-C1)./P1;  //Aplica Z-score na primeira coluna
ibase(:,2) = (ibase3(:,2)-C2)./P2;  //Aplica Z-score na segunda coluna
ibase(:,3) = (ibase3(:,3)-C3)./P3;  //Aplica Z-score na terceira coluna
ibase(:,4) = (ibase3(:,4)-C4)./P4;  //Aplica Z-score na quarta coluna
ibase(:,5) = (ibase3(:,5));     //Copia os dados da base para a quinta coluna
ibase(:,6) = (ibase3(:,6));     //Copia os dados da base para a sexta coluna
ibase(:,7) = (ibase3(:,7));     //Copia os dados da base para a sétima coluna

//PARTE 2: SOLICITA OS DADOS DO USUÁRIO

disp('---------------- ROBÔ BOTÂNICO USANDO KNN ------------------------------');
k = input('Quantos valores você gostaria de comparar? (k) -> ');
G = input('Qual a largura da sépala? (em cm) -> ');
H = input('Qual o comprimento da sépala? (em cm) -> ');
I = input('Qual a largura da pétala? (em cm) -> ');
J = input('Qual o comprimento da pétala? (em cm) -> ');

//Normaliza os dados da entrada
ponto_teste(1,1) = ((G-C1)/P1);
ponto_teste(1,2) = ((H-C2)/P2);
ponto_teste(1,3) = ((I-C3)/P3);
ponto_teste(1,4) = ((J-C4)/P4);

//PARTE 3: CALCULANDO A DISTANCIA EUCLIDIANA DOS PONTOS

tam = size(ibase, 1);   // numero de linhas da matriz

for i=1:tam
    linha = ibase(i,1:4);
    resultado(i,:) = linha-ponto_teste;
    resultado2 = resultado.^2;
    d_euclidiana(i,:) = sqrt(sum(resultado2(i,:)));
end

base2 = [d_euclidiana, ibase];  //Cria uma nova matriz com as distâncias euclidianas inseridas

//PARTE 4: SELECIONANDO OS K MENORES VALORES

for i=1:k
    [min_valor, min_linha] = min(base2(:,1));
    k_menores(i,:) = base2(min_linha,:);
    base2(min_linha,:) = [];
end

disp("Os "+ string(k) +" valores mais próximos da sua escolha na base de dados são:");
disp(k_menores);

//PARTE 5: CONTANDO O NÚMERO DE ENTRADAS 1 EM CADA COLUNA DO RESULTADO

[nb6, loc6] = members(1, k_menores(:,6));
[nb7, loc7] = members(1, k_menores(:,7));
[nb8, loc8] = members(1, k_menores(:,8));

//Verifica qual das 3 colunas possui mais entradas 1. Esta será a escolha do sistema
[w,y] = max(nb6, nb7, nb8);     

//PARTE FINAL: CLASSIFICANDO O TIPO DE FLOR DE ACORDO COM O RESULTADO

if y==1 then
    disp("A flor é do tipo setosa!");
elseif y==2 then
    disp("A flor é do tipo versicolor!");
elseif y==3 then
    disp("A flor é do tipo virginica!");
end
```
## Discussão dos resultados obtidos

Ao executar o arquivo [tr2_q2_knn.sce](tr2_q2_knn.sce) no Scilab, podemos verificar a seguinte sequência: 
* A solicitação do número de vizinhos mais próximos a serem analisados:
![tr2_q2_knn_1](https://user-images.githubusercontent.com/51038132/68555032-5cb87d00-040a-11ea-88b0-8f1c6b47af0f.png)

* Os dados da flor são solicitados:
![tr2_q2_knn_2](https://user-images.githubusercontent.com/51038132/68555034-5d511380-040a-11ea-90c7-92caea9b5975.png)
![tr2_q2_knn_3](https://user-images.githubusercontent.com/51038132/68555035-5d511380-040a-11ea-8b29-0454d22c8f45.png)
![tr2_q2_knn_4](https://user-images.githubusercontent.com/51038132/68555036-5de9aa00-040a-11ea-9302-dc8ef870b895.png)
![tr2_q2_knn_5](https://user-images.githubusercontent.com/51038132/68555037-5de9aa00-040a-11ea-9882-36bb224a36ec.png)

* A Matriz com os k elementos mais próximo é exibida e o tipo de flor classificado:
![tr2_q2_knn_6](https://user-images.githubusercontent.com/51038132/68555038-5de9aa00-040a-11ea-8f90-3b41f1569d8b.png)

### Parte 2: Extreme Learning Machine

### Iniciando 
Para a resolução desta questão e criação deste relatório foram usados os seguintes arquivos:

* [tr2_q2_ELM.sce](tr2_q2_ELM.sce) - Código-fonte da classificação usando ELM (Extreme Learning Machine)
* [iris_log.dat](iris_log.dat) - Conjunto de dados da questão

## Código comentado

```
// SEGUNDO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 2 (ELM)
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
//-----------------------------------------------------------------------------

clear;
clc;
base = fscanfMat('iris_log.dat');
X = base(:,1:4);
D = base(:,5:7);
q = 9; //qtd de neurônios ocultos
p = 4; //qtd atributos
a=0; b=0.1; // define intervalo dos pesos

x_ones  = ones(150,1);
X = [x_ones X];
X = X';
D = D';


//----------------leave-one-out--------------------------------------------
disp("método leave-one-out: \n");
W=a+(b-a).*rand(q,p+1); // gera numeros uniformes
YT1 = [];
for x = 1 : 150
    XT = X(:,x);
    DT = D(:,x);
    if (x==1)
        XTR = X(:,x+1 : 150);
        DTR = D(:,x+1 : 150);
      
    else
        XTR = X(:,[1:x-1, x+1 : 150]);
        DTR = D(:,[1:x-1, x+1 : 150]);
        
    end
    u1 = W*XTR;
    Z1 = 1./(1+exp(-u1));
    Z1 = [ones(1,149); Z1];
    M1 = DTR*Z1'*(Z1*Z1')^(-1);

    ut1 = W*XT;
    ZT1 = 1./(1+exp(-ut1));
    ZT1 = [ones(1,1); ZT1];
    ach = 0;
    YT1 = [YT1 M1*ZT1]; 
    
end

ach = 0;
for i = 1 : 150
    
    if(max(YT1(:,i)) == YT1(1,i))
        YT1(:,i) = [1; 0; 0];
    elseif(max(YT1(:,i)) == YT1(2,i))
        YT1(:,i) = [0; 1; 0];
    else
        YT1(:,i) = [0; 0; 1];
    end
    if(YT1(:,i) == D(:,i))
        ach = ach+1;
    end
end
// disp(YT1');
disp("Acuracia do método leave-one-out: ");
disp(ach/150);
```
## Discussão dos resultados obtidos

Ao executar o arquivo [tr2_q2_ELM.sce](tr2_q2_ELM.sce) no Scilab, teremos a execução do algoritmo ELM aplicado à base de dados iris_log. Foi utilizado o método leave one out e e ao fim da execução o console retorna a acurácia do método.

### Parte 3: Multi Layer Perceptron (MLP)

### Iniciando 
Para a resolução desta questão e criação deste relatório foram usados os seguintes arquivos:

* [tr2_q2_MLP.sce](tr2_q2_MLP.sce) - Código-fonte da classificação usando MLP (Multi Layer Perceptron)
* [iris_log.dat](iris_log.dat) - Conjunto de dados da questão

## Código comentado

```
// SEGUNDO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 2 (Perceptron Multi Camadas MLP)
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
// Para que esta aplicação rode apropriadamente deve ser instalado o 
// ANN Toolbox. (https://atoms.scilab.org/toolboxes/ANN_Toolbox)
//-----------------------------------------------------------------------------

clc;    //Limpa a tela
clear;  //Limpa as variáveis armazenadas anteriormente

// Assegura o mesmo ponto de início toda vez
rand('seed', 0);

// definição da rede
// 4 neurônios por rede, incluindo a entrada
// 4 neurônios na camada de entrada, 4 na camada oculta e 1 na camada de saída
N = [4,4,3];

//matriz de treinamento x deixando a primeira entrada de fora (leave-one-out)
base = fscanfMat('iris_log.dat');
x = base(2:150,1:4)';

//Saída desejada: 100 para classe 1, 010 para classe 2, 001 para classe 3
t = base(2:150,5:7)';

//Taxa de aprendizado e limite de erro tolerado pela rede
lp = [2.5, 0];

//Inicializa a matriz de pesos
W = ann_FF_init(N);

//Ciclos de treinamento
T = 100;

W = ann_FF_Std_online(x,t,N,W,lp,T);
//x é a matriz de treino t é a saída W são os pesos inicializados,
//N é a arquitetura da rede neural, lp é a taxa de aprendizado e
//T é o número de iterações

//Execução completa
retorno = ann_FF_run(x,N,W) //a rede N foi testada usando x como base de treino,
// e W como os pesos das conexões
```
## Discussão dos resultados obtidos

*IMPORTANTE: Para a correta execução deste código a toolbox ANN Toolbox deve ser instalada no SciLab. a toolBox pode ser encontrada no link: https://atoms.scilab.org/toolboxes/ANN_Toolbox.*
Ao executar o arquivo [tr2_q2_MLP.sce](tr2_q2_MLP.sce) no Scilab, a matriz *resultado* é criada como saída da rede mostrando o seu grau de adaptação.

### Parte 4: Perceptron

### Iniciando 
Para a resolução desta questão e criação deste relatório foram usados os seguintes arquivos:

* [tr2_q2_perceptron.sce](tr2_q2_perceptron.sce) - Código-fonte da classificação usando Perceptron
* [iris_log.dat](iris_log.dat) - Conjunto de dados da questão

## Código comentado

```
// SEGUNDO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 2 (Perceptron)
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
//-----------------------------------------------------------------------------

clc;    //Limpa a tela
clear;  //Limpa as variáveis armazenadas anteriormente

base = fscanfMat ('iris_log.dat');  //Carrega a base de dados
q = 3;                              //qtd de neurônios 
p = 4;                              //qtd atributos
n = 0.001;                          //taxa de aprendizagem
X = base(:,1:4);                    //Carrega os dados de entrada na variável X
D = base(:,5:7);                    //Carrega as saídas na variável D
x_ones = ones(150,1)*(-1);          //Cria um vetor de 1s (bias)
X = [x_ones X];                     //Insere o vetor de 1s na entrada (bias)
W = zeros(q,p+1);                   //Cria a matriz de pesos
X = X';                             //Transpõe a matriz de entrada
D = D';                             //Transpõe a matriz de saída

//Método leave-one-out
disp("método leave-one-out: ");
for x = 1 : 150
    if (x==1)
        lista = [x+1 : 150];
    else
        lista = [1:x-1, x+1 : 150];
    end
    for i = lista
        Y = W*X(:,i);
        E = D(:,i) - Y;
        A = (X(:,i))';
        W = W + n*E*A;
       
    end
    XT = X(:,x);
    YT(:,x) = W*XT;
end
ach = 0;
for i = 1 : 150
    
    if(max(YT(:,i)) == YT(1,i))
        YT(:,i) = [1; 0; 0];
    elseif(max(YT(:,i)) == YT(2,i))
        YT(:,i) = [0; 1; 0];
    else
        YT(:,i) = [0; 0; 1];
    end
    if(YT(:,i) == D(:,i))
        ach = ach+1;
    end
end

disp("Acurácia do método leave-one-out: ");
disp(ach/150);
```
## Discussão dos resultados obtidos

Ao executar o arquivo [tr2_q2_MLP.sce](tr2_q2_MLP.sce) no Scilab, rede é ativada e o console retorna o grau de acurácia usando o método one-leave-out.
