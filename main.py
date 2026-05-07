import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class OtimizadorUnimontes:
    def __init__(self, func_str, vars_str=['x1', 'x2']):
        self.vars = sp.symbols(vars_str)
        self.f_expr = sp.sympify(func_str)
        
        # O SymPy atua aqui: calculando derivadas parciais algebricamente
        self.grad_expr = [sp.diff(self.f_expr, v) for v in self.vars]
        self.hess_expr = [[sp.diff(g, v) for v in self.vars] for g in self.grad_expr]
        
        print("\n--- [SymPy] Processamento Algébrico Concluído ---")
        print(f"Função Objetivo: {self.f_expr}")
        print(f"Vetor Gradiente: {self.grad_expr}")
        print(f"Matriz Hessiana: {self.hess_expr}\n")
        
        # Transformando as equações do SymPy em funções velozes do NumPy
        self.f = sp.lambdify(self.vars, self.f_expr, 'numpy')
        self.grad = sp.lambdify(self.vars, self.grad_expr, 'numpy')
        self.hess = sp.lambdify(self.vars, self.hess_expr, 'numpy')

    def seccao_aurea(self, f_uni, a=0.0, b=2.0, tol=1e-4):
        """Otimização unidimensional (Razão Áurea)"""
        phi = (1 + 5**0.5) / 2
        resphi = 2 - phi
        
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1, f2 = f_uni(x1), f_uni(x2)
        
        while abs(b - a) > tol:
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + resphi * (b - a)
                f1 = f_uni(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = b - resphi * (b - a)
                f2 = f_uni(x2)
        return (a + b) / 2

    def verificar_parada_tecnica(self, hist_f, hist_x, hist_g, tol_geral=1e-3, criterio='todos'):
        """Verifica os critérios de parada de forma isolada ou conjunta"""
        if len(hist_f) < 6:
            return False, "Continuar"
            
        # 1. Estabilização da Função Objetivo
        if criterio in ['funcao', 'todos']:
            f_ultimos = hist_f[-6:]
            delta_f_total = max(hist_f) - min(hist_f) if max(hist_f) != min(hist_f) else 1.0
            if max(f_ultimos) - min(f_ultimos) < 0.001 * delta_f_total:
                return True, "Estabilização da Função Objetivo"
            
        # 2. Estabilização das Variáveis
        if criterio in ['variaveis', 'todos']:
            normas_x = [np.linalg.norm(x) for x in hist_x]
            x_ultimos = normas_x[-6:]
            delta_x_total = max(normas_x) - min(normas_x) if max(normas_x) != min(normas_x) else 1.0
            if max(x_ultimos) - min(x_ultimos) < 0.001 * delta_x_total:
                return True, "Estabilização das Variáveis de Otimização"
            
        # 3. Anulação do Vetor Gradiente
        if criterio in ['gradiente', 'todos']:
            g_ultimos = hist_g[-3:]
            m_max = max(hist_g)
            mg = max(g_ultimos)
            if mg < 0.001 * m_max or mg < tol_geral:
                return True, "Anulação do Vetor Gradiente"
            
        return False, "Continuar"

    def otimizar(self, x0, metodo='gradiente', max_iter=50, tol=1e-3, direcoes_pre_def=None, alpha_broyden=None, criterio_parada='todos'):
        xk = np.array(x0, dtype=float)
        historico_x = [xk.copy()]
        historico_f = [self.f(*xk)]
        normas_grad = [np.linalg.norm(self.grad(*xk))]
        
        n = len(x0)
        Hk = np.eye(n) 
        
        for k in range(max_iter):
            g = np.array(self.grad(*xk))
            
            # Verificação do Critério Escolhido
            parar, motivo = self.verificar_parada_tecnica(historico_f, historico_x, normas_grad, tol, criterio=criterio_parada)
            if parar:
                print(f"\n✅ Parada na iteração {k}. Motivo: {motivo}")
                break
            
            # Cálculo da Direção (dk) usando NumPy
            if metodo == 'aleatorio':
                if direcoes_pre_def is not None and k < len(direcoes_pre_def):
                    dk = np.array(direcoes_pre_def[k])
                else:
                    dk = np.random.normal(0, 1, n)
            elif metodo == 'gradiente':
                dk = -g
            elif metodo == 'newton':
                H = np.array(self.hess(*xk))
                try:
                    dk = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    dk = -np.linalg.pinv(H) @ g
            elif metodo == 'quasi-newton':
                dk = -Hk @ g
            
            # Otimização unidimensional (alpha)
            f_passo = lambda a: self.f(*(xk + a * dk))
            alpha = self.seccao_aurea(f_passo)
            
            x_next = xk + alpha * dk
            
            # Correção Família de Broyden (Quasi-Newton)
            if metodo == 'quasi-newton':
                sk = x_next - xk
                yk = np.array(self.grad(*x_next)) - g
                sk = sk.reshape(-1, 1)
                yk = yk.reshape(-1, 1)
                
                if np.dot(yk.T, sk) > 1e-8:
                    C_DFP = (sk @ sk.T) / (sk.T @ yk) - (Hk @ yk @ yk.T @ Hk) / (yk.T @ Hk @ yk)
                    
                    # CORREÇÃO DA SINTAXE APLICADA AQUI (Inserido o operador @ que faltava)
                    term1 = 1 + (yk.T @ Hk @ yk) / (yk.T @ sk)
                    term2 = (sk @ sk.T) / (sk.T @ yk)
                    term3 = (sk @ yk.T @ Hk + Hk @ yk @ sk.T) / (yk.T @ sk)
                    C_BFGS = term1 * term2 - term3
                    
                    a_b = alpha_broyden[k] if (alpha_broyden is not None and k < len(alpha_broyden)) else 1.0
                    Ck = (1 - a_b) * C_DFP + a_b * C_BFGS
                    Hk = Hk + Ck
            
            xk = x_next
            historico_x.append(xk.copy())
            historico_f.append(self.f(*xk))
            normas_grad.append(np.linalg.norm(self.grad(*xk)))
            
        return np.array(historico_x), np.array(historico_f), np.array(normas_grad)

    def plotar_graficos_lista(self, hist_x, hist_f, hist_g):
        """Gera os gráficos com Curvas de Nível exigidos nas Atividades"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Curva de Convergência da Função Objetivo
        ax1 = fig.add_subplot(221)
        ax1.plot(hist_f, 'b-o', lw=2)
        ax1.set_title("Curva de Convergência (Função Objetivo)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Iterações")
        ax1.set_ylabel("Valor f(x)")
        ax1.grid(True, ls="--")
        
        # 2. Curva de Deslocamento com CURVAS DE NÍVEL
        ax2 = fig.add_subplot(222)
        
        margem_x1 = abs(max(hist_x[:, 0]) - min(hist_x[:, 0])) * 0.5 + 1
        margem_x2 = abs(max(hist_x[:, 1]) - min(hist_x[:, 1])) * 0.5 + 1
        x1_vals = np.linspace(min(hist_x[:, 0]) - margem_x1, max(hist_x[:, 0]) + margem_x1, 100)
        x2_vals = np.linspace(min(hist_x[:, 1]) - margem_x2, max(hist_x[:, 1]) + margem_x2, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.zeros_like(X1)
        
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = self.f(X1[i, j], X2[i, j])
                
        contour = ax2.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
        
        ax2.plot(hist_x[:, 0], hist_x[:, 1], 'k--', alpha=0.8)
        sc = ax2.scatter(hist_x[:, 0], hist_x[:, 1], c=range(len(hist_x)), cmap='coolwarm', s=60, zorder=5)
        ax2.plot(hist_x[0, 0], hist_x[0, 1], 'gs', ms=10, label="X0 (Início)")
        ax2.plot(hist_x[-1, 0], hist_x[-1, 1], 'r*', ms=15, label="X* (Final)")
        ax2.set_title("Curva de Deslocamento + Curvas de Nível", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Variável x1")
        ax2.set_ylabel("Variável x2")
        ax2.legend()
        ax2.grid(True, ls="--")
        
        # 3. Curva de Convergência das Variáveis
        ax3 = fig.add_subplot(223)
        ax3.plot(hist_x[:, 0], 'g-^', label="x1", lw=2)
        ax3.plot(hist_x[:, 1], 'm-v', label="x2", lw=2)
        ax3.set_title("Curva de Convergência das Variáveis x1 e x2", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Iterações")
        ax3.set_ylabel("Valor das Variáveis")
        ax3.legend()
        ax3.grid(True, ls="--")
        
        # 4. Curva de Convergência do Gradiente
        ax4 = fig.add_subplot(224)
        ax4.plot(hist_g, 'r-s', lw=2)
        ax4.set_yscale('log')
        ax4.set_title("Curva de Convergência do Módulo do Gradiente", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Iterações")
        ax4.set_ylabel("||∇f(x)|| (Log)")
        ax4.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        plt.show()

# ==========================================
# MENU MASTER - LISTA & ATIVIDADE
# ==========================================
if __name__ == "__main__":
    print("=== OTIMIZADOR UNIVERSAL ===")
    print("1. Lista de Exercícios (ONL-I/2026)")
    print("2. Atividade Avaliativa 02 (2024/01)")
    
    tipo_ativ = input("Escolha o módulo (1 ou 2): ").strip()
    
    # Valores default
    func_padrao = "x1**2 + x2**2"
    x_init_padrao = [0.0, 0.0]
    metodo_padrao = 'gradiente'
    parada_padrao = 'todos'
    direcoes = None
    alfas_q = None
    max_i = 50

    if tipo_ativ == '1':
        print("\n--- Lista de Exercícios ---")
        print("1. Ex 1 (Booth / Gradiente)")
        print("2. Ex 2 (Booth / Newton)")
        print("3. Ex 3 (Booth / Aleatório Pré-Definido)")
        print("4. Ex 4 (Camel / Newton Modificado)")
        print("5. Ex 5 (Camel / Quasi-Newton com Alfas)")
        escolha = input("Escolha a questão (1-5): ").strip()
        
        if escolha in ['1', '2', '3']:
            func_padrao = "(x1+2*x2-7)**2 + (2*x1+x2-5)**2"
            x_init_padrao = [0.5, 3.5]
        elif escolha in ['4', '5']:
            func_padrao = "2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2"
            x_init_padrao = [0.5, 0.5]
            
        if escolha == '1':
            metodo_padrao = 'gradiente'; max_i = 5
        elif escolha == '2':
            metodo_padrao = 'newton'; max_i = 10
        elif escolha == '3':
            metodo_padrao = 'aleatorio'; max_i = 5
            direcoes = [[-0.3850, 0.6680], [0.4560, -0.4956], [0.8570, 0.1049], [0.4956, -0.3456], [0.6680, 0.7829]]
        elif escolha == '4':
            metodo_padrao = 'newton'; max_i = 15
        elif escolha == '5':
            metodo_padrao = 'quasi-newton'; max_i = 15
            alfas_q = [0.4125, 0.7530]

    elif tipo_ativ == '2':
        print("\n--- Atividade Avaliativa 02 ---")
        print("3. Questão 3 e 4 (Problema A / Gradiente / Parada Gradiente)")
        print("5. Questão 5 (Dixon-Price / Newton ou Outro)")
        print("9. Questão 9 (Camel / Quasi-Newton / Parada Variáveis)")
        escolha = input("Escolha a questão (3, 5, 9): ").strip()
        XX = input("Digite o valor 'XX' da sua matrícula (ex: 7.7): ").strip()
        if not XX: XX = "0.0"
        
        if escolha == '3':
            func_padrao = "(x1 - 3)**2 / 4 + (x2 - 2)**2 / 9 + 13"
            x_init_padrao = [float(XX), 5.0]
            metodo_padrao = 'gradiente'
            parada_padrao = 'gradiente'
        elif escolha == '5':
            func_padrao = "(x1 - 1)**2 + 2*(2*x2**2 - x1)**2"
            x_init_padrao = [-float(XX), -10.0]
            metodo_padrao = 'newton' 
            parada_padrao = 'todos'
        elif escolha == '9':
            func_padrao = "2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2"
            x_init_padrao = [2.0, float(XX)]
            metodo_padrao = 'quasi-newton'
            parada_padrao = 'variaveis'

    print(f"\n[Padrão carregado para a sua escolha]")
    print(f"Equação Padrão: {func_padrao}")
    print(f"Ponto X0 Padrão: {x_init_padrao}")
    
    # === ENTRADA INTERATIVA GERAL ===
    func_input = input("\nDigite a nova equação (ou pressione ENTER para usar a padrão): ").strip()
    func_final = func_input if func_input else func_padrao
    
    x_input = input("Digite o novo ponto inicial separado por vírgula (ou ENTER para usar o padrão): ").strip()
    if x_input:
        x_init_final = [float(x.strip()) for x in x_input.split(',')]
    else:
        x_init_final = x_init_padrao
        
    print(f"\n[Configuração Final da Execução]")
    print(f"Equação: {func_final}")
    print(f"Ponto X0: {x_init_final}")
    print(f"Método: {metodo_padrao} | Parada: {parada_padrao}")
    
    opt = OtimizadorUnimontes(func_final)
    hx, hf, hg = opt.otimizar(x_init_final, metodo=metodo_padrao, max_iter=max_i, 
                              criterio_parada=parada_padrao, direcoes_pre_def=direcoes, alpha_broyden=alfas_q)
        
    print("\n[!] Resultados Obtidos:")
    print(f"-> Mínimo encontrado: {hx[-1]}")
    print(f"-> f(x) no mínimo: {hf[-1]:.6f}")
    
    opt.plotar_graficos_lista(hx, hf, hg)