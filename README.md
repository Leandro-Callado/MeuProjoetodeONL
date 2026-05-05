# Otimizador Não Linear 

Este projeto contém um otimizador numérico em Python para resolver exercícios de otimização unimontes, incluindo métodos de gradiente, Newton, quasi-Newton e busca unidimensional pela razão áurea.

## Arquivos

- `main.py` - Implementação principal do otimizador e menu interativo para executar diferentes exercícios de otimização.
- `require.txt` - Lista de dependências necessárias para executar o projeto.

## Dependências

As bibliotecas necessárias são:

- `numpy`
- `matplotlib`
- `sympy`

Instale-as com:

```bash
pip install -r require.txt
```

## Uso

Execute o script principal:

```bash
python main.py
```

Ao iniciar, o programa apresenta um menu com cinco exercícios:

1. Exercício 1 - Método do gradiente para função de Booth
2. Exercício 2 - Método de Newton para função de Booth
3. Exercício 3 - Direções pré-definidas / método aleatório para função de Booth
4. Exercício 4 - Método de Newton para função de Camel
5. Exercício 5 - Método quasi-Newton com correção de Broyden para função de Camel

Você pode escolher um exercício e, em seguida, informar uma nova função ou usar a função padrão. Também é possível fornecer um ponto inicial personalizado.

## Recursos

- Otimização com métodos de direção de descenso
- Busca unidimensional pela razão áurea
- Critérios de parada baseados em evolução da função objetivo, estabilidade das variáveis e módulo do gradiente
- Geração de gráficos de convergência usando `matplotlib`

## Observações

- O script usa `sympy` para simbolicamente diferenciar a função e gerar gradiente e Hessiana.
- O resultado final mostra o ponto mínimo encontrado, o valor da função no mínimo e exibe gráficos de convergência.

## Licença

Este projeto é licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
