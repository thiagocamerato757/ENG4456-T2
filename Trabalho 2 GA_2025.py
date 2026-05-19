import numpy as np
import math
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Callbacks
from geneticalgorithm2 import Population_initializer
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
 
# ==========================================================================================
# FUNÇÃO OBJETIVO — Rastrigin 2D
# ==========================================================================================
 
def f(X):
    dim = len(X)
    OF = 0
    for i in range(dim):
        OF += (X[i]**2) - 10 * math.cos(2 * math.pi * X[i]) + 10
    return OF
 
# ==========================================================================================
# PARÂMETROS BASELINE DO GA
# ==========================================================================================
 
varbound = np.array([[-5.12, 5.12]] * 2)
 
max_num_iteration    = 300
population_size      = 100
mutation_probability = 0.1
elit_ratio           = 0.01
crossover_probability= 0.5
parents_portion      = 0.3
crossover_type       = 'one_point'
selection_type       = 'roulette'
 
num_experimentos = 20  # rodadas por configuração
 
algorithm_param = {
    'max_num_iteration':         max_num_iteration,
    'population_size':           population_size,
    'mutation_probability':      mutation_probability,
    'elit_ratio':                elit_ratio,
    'crossover_probability':     crossover_probability,
    'parents_portion':           parents_portion,
    'crossover_type':            crossover_type,
    'max_iteration_without_improv': None,
    'mutation_type':             'uniform_by_center',
    'selection_type':            selection_type,
}
 
# ==========================================================================================
# CLASSE PARA EXPERIMENTOS COMPARATIVOS
# ==========================================================================================
 
class ExperimentComparison:
    """
    Executa e organiza experimentos variando UM parâmetro do GA por vez.
    Cada rodada cria um novo objeto `ga` (sem reuso de estado).
    """
 
    def __init__(self, objective_function, variable_boundaries, baseline_params, num_experiments=20):
        self.f            = objective_function
        self.varbound     = variable_boundaries
        self.baseline     = deepcopy(baseline_params)
        self.num_experiments = num_experiments
        self.results      = {}  # {param_name: {param_value: data}}
 
    # --------------------------------------------------------------------------------------
    def run_experiment_set(self, param_name, param_values, verbose=True):
        """
        Executa `num_experiments` rodadas para cada valor em `param_values`,
        variando apenas `param_name` em relação ao baseline.
        """
        self.results[param_name] = {}
 
        for param_value in param_values:
            if verbose:
                print(f"\n{'='*70}")
                print(f"Testando {param_name} = {param_value}")
                print(f"{'='*70}")
 
            params = deepcopy(self.baseline)
            params[param_name] = param_value
 
            convergences = []
            for simu in range(self.num_experiments):
                if verbose:
                    print(f"  Rodada {simu+1}/{self.num_experiments}...", end='\r')
 
                # Novo objeto a cada rodada — evita reuso de estado (sem viés)
                model = ga(
                    function=self.f,
                    dimension=2,
                    variable_type='real',
                    variable_boundaries=self.varbound,
                    algorithm_parameters=params,
                )
 
                model.run(
                    no_plot=True,
                    disable_progress_bar=True,
                    studEA=True,
                    remove_duplicates_generation_step=2,
                    population_initializer=Population_initializer(
                        select_best_of=1,
                        local_optimization_step='never',
                        local_optimizer=None,
                    ),
                )
 
                convergences.append(model.report)
 
            if verbose:
                print(f"  Rodada {self.num_experiments}/{self.num_experiments}... Concluído!")
 
            mean_convergence = self._calculate_mean_convergence(convergences)
 
            self.results[param_name][param_value] = {
                'convergences':    convergences,
                'mean_convergence': mean_convergence,
                'final_value':     mean_convergence[-1],
                'num_generations': len(mean_convergence),
                'std_final':       np.std( [c[-1] for c in convergences]),
                'min_final':       np.min( [c[-1] for c in convergences]),
                'max_final':       np.max( [c[-1] for c in convergences]),
            }
 
    # --------------------------------------------------------------------------------------
    def _calculate_mean_convergence(self, convergences):
        """Média geração-a-geração das `convergences` coletadas."""
        # Use minimum length to handle runs with different termination points
        num_gen = min(len(c) for c in convergences)
        return [
            np.mean([convergences[j][i] for j in range(len(convergences))])
            for i in range(num_gen)
        ]
 
    # --------------------------------------------------------------------------------------
    def plot_comparison(self, param_name, figsize=(14, 5)):
        """Curva de convergência comparativa + barra de resultado final."""
        if param_name not in self.results:
            print(f"Parâmetro '{param_name}' não encontrado nos resultados.")
            return
 
        data         = self.results[param_name]
        param_values = sorted(data.keys(), key=lambda x: (x is None, x) )

        colors       = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
 
        fig, axes = plt.subplots(1, 2, figsize=figsize)
 
        # — Gráfico 1: convergência média
        ax1 = axes[0]
        for idx, pv in enumerate(param_values):
            ax1.plot(data[pv]['mean_convergence'],
                     label=f"{param_name}={pv}",
                     color=colors[idx], linewidth=2)
        ax1.set_xlabel('Geração', fontsize=12)
        ax1.set_ylabel('Melhor Fitness f(x)', fontsize=12)
        ax1.set_title(f'Convergência: Efeito de {param_name}', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
 
        # — Gráfico 2: resultado final com desvio padrão
        ax2 = axes[1]
        final_values = [data[pv]['final_value'] for pv in param_values]
        stds         = [data[pv]['std_final']   for pv in param_values]
        ax2.bar(range(len(param_values)), final_values,
                color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.errorbar(range(len(param_values)), final_values, yerr=stds,
                     fmt='none', color='black', capsize=5, capthick=2)
        ax2.set_xlabel(param_name, fontsize=12)
        ax2.set_ylabel('f(x) mínimo encontrado', fontsize=12)
        ax2.set_title('Qualidade da Solução Final', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(param_values)))
        ax2.set_xticklabels(param_values)
        ax2.grid(True, alpha=0.3, axis='y')
 
        plt.tight_layout()
        plt.savefig(f'comparison_{param_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
 
    # --------------------------------------------------------------------------------------
    def plot_all_convergences(self, param_name, figsize=(14, 6)):
        """Todas as rodadas individuais + média, um subplot por valor testado."""
        if param_name not in self.results:
            print(f"Parâmetro '{param_name}' não encontrado nos resultados.")
            return
 
        data         = self.results[param_name]
        param_values = sorted(data.keys(), key=lambda x: (x is None, x) )
 
        fig, axes = plt.subplots(1, len(param_values),
                                 figsize=(5 * len(param_values), 5))
        if len(param_values) == 1:
            axes = [axes]
 
        for idx, pv in enumerate(param_values):
            ax          = axes[idx]
            convergences = data[pv]['convergences']
            colors      = plt.cm.viridis(np.linspace(0, 1, len(convergences)))
 
            for i, conv in enumerate(convergences):
                ax.plot(conv, color=colors[i], alpha=0.7,
                        linewidth=1.5, label=f'Rodada {i+1}')
 
            ax.plot(data[pv]['mean_convergence'],
                    color='red', linewidth=3, linestyle='--', label='Média')
 
            ax.set_xlabel('Geração', fontsize=11)
            ax.set_ylabel('Melhor Fitness f(x)', fontsize=11)
            ax.set_title(f'{param_name} = {pv}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
 
        plt.tight_layout()
        plt.savefig(f'all_convergences_{param_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
 
    # --------------------------------------------------------------------------------------
    def get_report_table(self, param_name):
        """Retorna DataFrame com estatísticas por valor testado."""
        if param_name not in self.results:
            print(f"Parâmetro '{param_name}' não encontrado nos resultados.")
            return None
 
        data         = self.results[param_name]
        # Sort with None values placed last
        param_values = sorted(data.keys(), key=lambda x: (x is None, x) )
 
        rows = []
        for pv in param_values:
            rows.append({
                param_name:       pv,
                'Gerações':       data[pv]['num_generations'],
                'f(x) mínimo':    data[pv]['final_value'],
                'Desvio Padrão':  data[pv]['std_final'],
                'Mínimo (melhor)':data[pv]['min_final'],
                'Máximo (pior)':  data[pv]['max_final'],
            })
        return pd.DataFrame(rows)
 
    def get_report_table_display(self, param_name):
        """Retorna DataFrame arredondado para exibição legível."""
        df = self.get_report_table(param_name)
        if df is not None:
            df_display = df.copy()
            # Arredondar apenas as colunas de ponto flutuante para exibição
            for col in ['f(x) mínimo', 'Desvio Padrão', 'Mínimo (melhor)', 'Máximo (pior)']:
                df_display[col] = df_display[col].round(6)
            return df_display
        return None
 
    # --------------------------------------------------------------------------------------
    def print_report(self, param_name):
        """Imprime e salva em CSV a tabela de resultados."""
        df_display = self.get_report_table_display(param_name)
        df_full = self.get_report_table(param_name)
        
        if df_display is not None:
            print(f"\n{'='*80}")
            print(f"RELATÓRIO: {param_name}")
            print(f"{'='*80}")
            
            # Formatar para exibição sem notação científica
            with pd.option_context('display.float_format', '{:.10f}'.format):
                print(df_display.to_string(index=False))
            
            print(f"{'='*80}\n")
            
            # Salvar dados BRUTOS (sem arredondamento) em CSV
            csv_filename = f'report_{param_name}.csv'
            df_full.to_csv(csv_filename, index=False, float_format='%.15g')
            print(f"✓ Tabela completa (precisão total) salva em: {csv_filename}\n")