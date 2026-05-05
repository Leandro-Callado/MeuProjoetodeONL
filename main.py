import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class OtimizadorUnimontes:
    def __init__(self, func_str, vars_str=['x1', 'x2']):
        self.vars = sp.symbols(vars_str)
        self.f_expr = sp.sympify(func_str)
        
        self.f = sp.lambdify(self.vars, self.f_expr, 'numpy')
        self.grad_expr = [sp.diff(self.f_expr, v) for v in self.vars]
        self.grad = sp.lambdify(self.vars, self.grad_expr, 'numpy')
        
        hess_expr = [[sp.diff(g, v) for v in self.vars] for g in self.grad_expr]
        self.hess = sp.lambdify(self.vars, hess_expr, 'numpy')

    def seccao_aurea(self, f_uni, a=0.0, b=2.0, tol=1e-4):
        """Otimização unidimensional (Razão Áurea)[cite: 4]"""
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

    def verificar_parada_tecnica(self, hist_f, hist_x, hist_g, tol_geral=1e-3):
        """Implementa os 3 critérios de parada do material[cite: 5]"""
        if len(hist_f) < 6:
            return False, "Continuar"
            
        # 1. Estabilização da Função Objetivo nas últimas 5 iterações[cite: 5]
        f_ultimos = hist_f[-6:]
        f_max, f_min = max(hist_f), min(hist_f)
        delta_f_total = f_max - f_min if f_max != f_min else 1.0
        delta_f_5 = max(f_ultimos) - min(f_ultimos)
        if delta_f_5 < 0.001 * delta_f_total:
            return True, "Estabilização da Função Objetivo"
            
        # 2. Estabilização das Variáveis (Norma)[cite: 5]
        normas_x = [np.linalg.norm(x) for x in hist_x]
        x_ultimos = normas_x[-6:]
        delta_x_total = max(normas_x) - min(normas_x) if max(normas_x) != min(normas_x) else 1.0
        delta_x_5 = max(x_ultimos) - min(x_ultimos)
        if delta_x_5 < 0.001 * delta_x_total:
            return True, "Estabilização das Variáveis de Otimização"
            
        # 3. Anulação do Vetor Gradiente[cite: 5]
        g_ultimos = hist_g[-3:]
        m_max = max(hist_g)
        mg = max(g_ultimos)
        if mg < 0.001 * m_max or mg < tol_geral:
            return True, "Anulação do Vetor Gradiente"
            
        return False, "Continuar"

    def otimizar(self, x0, metodo='gradiente', max_iter=50, tol=1e-3, direcoes_pre_def=None, alpha_broyden=None):
        xk = np.array(x0, dtype=float)
        historico_x = [xk.copy()]
        historico_f = [self.f(*xk)]
        normas_grad = [np.linalg.norm(self.grad(*xk))]
        
        n = len(x0)
        Hk = np.eye(n) # Inicia Quasi-Newton com Identidade[cite: 3]
        
        for k in range(max_iter):
            g = np.array(self.grad(*xk))
            
            # Verificação de Critérios de Parada[cite: 5]
            parar, motivo = self.verificar_parada_tecnica(historico_f, historico_x, normas_grad, tol)
            if parar:
                print(f"\n✅ Parada na iteração {k}. Motivo: {motivo}")
                break
            
            # Cálculo da Direção (dk)[cite: 1, 2, 3]
            if metodo == 'aleatorio':
                if direcoes_pre_def is not None and k < len(direcoes_pre_def):
                    dk = np.array(direcoes_pre_def[k]) # Permite resolver Ex 3[cite: 7]
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
            
            # Correção Família de Broyden para Quasi-Newton[cite: 3]
            if metodo == 'quasi-newton':
                sk = x_next - xk
                yk = np.array(self.grad(*x_next)) - g
                sk = sk.reshape(-1, 1)
                yk = yk.reshape(-1, 1)
                
                # Formulações DFP e BFGS[cite: 3]
                if np.dot(yk.T, sk) > 1e-8:
                    # DFP
                    C_DFP = (sk @ sk.T) / (sk.T @ yk) - (Hk @ yk @ yk.T @ Hk) / (yk.T @ Hk @ yk)
                    
                    # BFGS
                    term1 = 1 + (yk.T @ Hk @ yk) / (yk.T @ sk)
                    term2 = (sk @ sk.T) / (sk.T @ yk)
                    term3 = (sk @ yk.T @ Hk + Hk @ yk @ sk.T) / (yk.T @ sk)
                    C_BFGS = term1 * term2 - term3
                    
                    # Mistura de Broyden (Ex 5)[cite: 3, 7]
                    a_b = alpha_broyden[k] if (alpha_broyden is not None and k < len(alpha_broyden)) else 1.0 # 1.0 = BFGS puro
                    Ck = (1 - a_b) * C_DFP + a_b * C_BFGS
                    Hk = Hk + Ck
            
            xk = x_next
            historico_x.append(xk.copy())
            historico_f.append(self.f(*xk))
            normas_grad.append(np.linalg.norm(self.grad(*xk)))
            
        return np.array(historico_x), np.array(historico_f), np.array(normas_grad)

    def plotar_graficos_lista(self, hist_x, hist_f, hist_g):
        """Gera EXATAMENTE os gráficos solicitados na Lista de Exercícios[cite: 7]"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Curva de Convergência da Função Objetivo[cite: 7]
        ax1 = fig.add_subplot(221)
        ax1.plot(hist_f, 'b-o', lw=2)
        ax1.set_title("Curva de Convergência (Função Objetivo)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Iterações")
        ax1.set_ylabel("Valor f(x)")
        ax1.grid(True, ls="--")
        
        # 2. Curva de Deslocamento (x1 vs x2)[cite: 7]
        ax2 = fig.add_subplot(222)
        ax2.plot(hist_x[:, 0], hist_x[:, 1], 'k--', alpha=0.5)
        sc = ax2.scatter(hist_x[:, 0], hist_x[:, 1], c=range(len(hist_x)), cmap='coolwarm', s=60, zorder=5)
        ax2.plot(hist_x[0, 0], hist_x[0, 1], 'gs', ms=10, label="X0 (Início)")
        ax2.plot(hist_x[-1, 0], hist_x[-1, 1], 'r*', ms=15, label="X* (Final)")
        ax2.set_title("Curva de Deslocamento do Método", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Variável x1")
        ax2.set_ylabel("Variável x2")
        ax2.legend()
        ax2.grid(True, ls="--")
        
        # 3. Curva de Convergência das Variáveis[cite: 7]
        ax3 = fig.add_subplot(223)
        ax3.plot(hist_x[:, 0], 'g-^', label="x1", lw=2)
        ax3.plot(hist_x[:, 1], 'm-v', label="x2", lw=2)
        ax3.set_title("Curva de Convergência das Variáveis x1 e x2", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Iterações")
        ax3.set_ylabel("Valor das Variáveis")
        ax3.legend()
        ax3.grid(True, ls="--")
        
        # 4. Curva de Convergência do Gradiente (Adicional para Análise)[cite: 7]
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
# GABARITO DE EXECUÇÃO RÁPIDA (Com base na Lista)
# ==========================================
if __name__ == "__main__":
    print("--- Otimizador Resolutor da Lista ONL-I/2026 ---")
    print("1. Exercício 1 (Booth / Gradiente)")
    print("2. Exercício 2 (Booth / Newton)")
    print("3. Exercício 3 (Booth / Aleatório Pré-Definido)")
    print("4. Exercício 4 (Camel / Newton Modificado)")
    print("5. Exercício 5 (Camel / Quasi-Newton com Alfas)")
    
    escolha = input("\nQual exercício deseja rodar? (1-5): ")
    
    # Definindo os padrões de acordo com a escolha do menu[cite: 7]
    if escolha in ['1', '2', '3']:
        func_padrao = "(x1+2*x2-7)**2 + (2*x1+x2-5)**2"
        x_init_padrao = [0.5, 3.5]
    elif escolha in ['4', '5']:
        func_padrao = "2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2"
        x_init_padrao = [0.5, 0.5]
    else:
        print("Opção inválida.")
        exit()
        
    print(f"\n[Equação Padrão do Exercício {escolha}]: {func_padrao}")
    func_input = input("Digite a nova equação (ou pressione ENTER para usar a padrão): ").strip()
    func = func_input if func_input else func_padrao
    
    print(f"\n[Ponto Inicial Padrão]: {x_init_padrao}")
    x_input = input("Digite o novo ponto inicial separado por vírgula (ou ENTER para usar o padrão): ").strip()
    if x_input:
        x_init = [float(x.strip()) for x in x_input.split(',')]
    else:
        x_init = x_init_padrao

    opt = OtimizadorUnimontes(func)
    
    # Configurando Parâmetros por Exercício
    if escolha == '1':
        hx, hf, hg = opt.otimizar(x_init, metodo='gradiente', max_iter=5)
    elif escolha == '2':
        hx, hf, hg = opt.otimizar(x_init, metodo='newton', max_iter=10)
    elif escolha == '3':
        dirs_ex3 = [[-0.3850, 0.6680], [0.4560, -0.4956], [0.8570, 0.1049], [0.4956, -0.3456], [0.6680, 0.7829]][cite: 7]
        hx, hf, hg = opt.otimizar(x_init, metodo='aleatorio', max_iter=5, direcoes_pre_def=dirs_ex3)
    elif escolha == '4':
        hx, hf, hg = opt.otimizar(x_init, metodo='newton', max_iter=15, tol=1e-3)
    elif escolha == '5':
        alfas = [0.4125, 0.7530][cite: 7]
        hx, hf, hg = opt.otimizar(x_init, metodo='quasi-newton', max_iter=15, alpha_broyden=alfas)
        
    print("\n[!] Resultados Obtidos:")
    print(f"-> Mínimo encontrado: {hx[-1]}")
    print(f"-> f(x) no mínimo: {hf[-1]:.6f}")
    
    opt.plotar_graficos_lista(hx, hf, hg)