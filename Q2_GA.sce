// TERCEIRO TRABALHO DE INTELIGÊNCIA COMPUTACIONAL
// Questão 2
// Aluno: José Lopes de Souza Filho
// Matrícula: 389097
// Aplicação: Scilab, versão 6.0.2
// SO: Linux Mint 19.2 Tina
//-----------------------------------------------------------------------------

clear;
clc;
p = grand(100,20, "uin", 0, 1); //gera uma matriz binaria 100x20
ant = 0;
passo = 20/1023; // valor correspondente a um bit
for q = 1:1000 //quantidade de gerações
    nota = [];
    
    for i = 1:100
        x=0;
        y=0;
        //conversão da representação binária para real
        for j = 1:10
            if p(i,j)==1
                x = x +(2^(10-j)*passo);
            end
            if p(i,j+10)==1
                y = y +(2^(10-j)*passo);
            end
        end
        a = abs(x*sin(y*%pi/4)+y*sin(x*%pi/4));
        if a > ant
            xm = x;
            ym = y;
        end
        if ant < a
            ant = a;
        end
        nota = [nota; a]; 
        
    end
    
    
   
    p = [p nota];

    p = gsort(p,'r');

    pais = p(1:50,1:20);
    p = [];
    for i = 1:2:100
       pai1 = grand(1,1,"uin",1,50);
       pai2 = grand(1,1,"uin",1,50);
       xcross = grand(1,1,"uin",1,20);
       p(i,:) = [pais(pai1,1:xcross) pais(pai2,xcross+1:20)];
       p(i+1,:) = [pais(pai2,1:xcross) pais(pai1,xcross+1:20)];
    end
end

disp(ant);
disp("valor de x e y é: ");
disp(xm);
disp(" e ");
disp(ym);
