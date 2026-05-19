# -*- coding: utf-8 -*-
"""Comparação completa entre baseline e configuração otimizada (corrigido)."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Population_initializer

# =============================================================================
# Função objetivo (Rastrigin 2D)
# =============================================================================
def f(X):
    dim = len(X)
    return sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10 for x in X)

varbound = np.array([[-5.12, 5.12]] * 2)

# =============================================================================
# Configuração baseline (igual à Seção 3.1)
# =============================================================================
baseline_params = {
    'max_num_iteration': 300,
    'population_size': 100,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'one_point',
    'selection_type': 'roulette',
    'max_iteration_without_improv': None,
    'mutation_type': 'uniform_by_center',
}

# =============================================================================
# Configuração otimizada (melhores parâmetros encontrados)
# =============================================================================
optimized_params = {
    'max_num_iteration': 300,
    'population_size': 200,
    'mutation_probability': 0.3,
    'elit_ratio': 0.1,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'selection_type': 'tournament',
    'max_iteration_without_improv': 100,
    'mutation_type': 'uniform_by_center',
}

# =============================================================================
# Função para rodar múltiplas execuções e extrair estatísticas
# =============================================================================
def run_multiple(config, num_runs=20, verbose=False):
    best_vars = []
    best_vals = []
    all_convergences = []
    
    for i in range(num_runs):
        if verbose:
            print(f"Execução {i+1}/{num_runs}...", end='\r')
        
        model = ga(
            function=f,
            dimension=2,
            variable_type='real',
            variable_boundaries=varbound,
            algorithm_parameters=config,
        )
        # O método run() retorna um dicionário com os resultados
        output = model.run(
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
        # A melhor solução está no dicionário de retorno
        best_vars.append(output.variable)
        best_vals.append(output.function)
        all_convergences.append(model.report)  # histórico ainda está no objeto
    
    if verbose:
        print(f"Execução {num_runs}/{num_runs} concluída!")
    
    # Média das curvas de convergência (considerando o menor comprimento)
    min_len = min(len(c) for c in all_convergences)
    mean_conv = [np.mean([c[i] for c in all_convergences]) for i in range(min_len)]
    
    return {
        'best_vars': best_vars,
        'best_vals': best_vals,
        'mean_conv': mean_conv,
        'std_final': np.std(best_vals),
        'mean_final': np.mean(best_vals),
        'min_final': np.min(best_vals),
        'max_final': np.max(best_vals),
    }

# =============================================================================
# Executar ambos os modelos
# =============================================================================
print("="*60)
print("Executando Baseline (20 rodadas)...")
print("="*60)
baseline_results = run_multiple(baseline_params, num_runs=20, verbose=True)

print("\n" + "="*60)
print("Executando Configuração Otimizada (20 rodadas)...")
print("="*60)
optimized_results = run_multiple(optimized_params, num_runs=20, verbose=True)

# =============================================================================
# Tabela comparativa
# =============================================================================
df_compare = pd.DataFrame({
    'Configuração': ['Baseline', 'Otimizada'],
    'f(x) médio': [baseline_results['mean_final'], optimized_results['mean_final']],
    'Desvio Padrão': [baseline_results['std_final'], optimized_results['std_final']],
    'Melhor f(x)': [baseline_results['min_final'], optimized_results['min_final']],
    'Pior f(x)': [baseline_results['max_final'], optimized_results['max_final']],
    'Gerações (média)': [len(baseline_results['mean_conv']), len(optimized_results['mean_conv'])],
})

print("\n" + "="*60)
print("TABELA COMPARATIVA")
print("="*60)
print(df_compare.round(8).to_string(index=False))

# =============================================================================
# Gráfico de convergência comparativo
# =============================================================================
plt.figure(figsize=(10, 6))
plt.plot(baseline_results['mean_conv'], label='Baseline', color='blue', linewidth=2)
plt.plot(optimized_results['mean_conv'], label='Otimizada', color='red', linewidth=2)
plt.xlabel('Geração', fontsize=12)
plt.ylabel('Melhor Fitness f(x)', fontsize=12)
plt.title('Comparação de Convergência: Baseline vs. Configuração Otimizada', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('comparacao_baseline_otimizada.png', dpi=150)
plt.show()

# =============================================================================
# Resultados finais (coordenadas x* e f(x*))
# =============================================================================
print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"Baseline       -> melhor x* = [{baseline_results['best_vars'][0][0]:.10f}, {baseline_results['best_vars'][0][1]:.10f}], f(x*) = {baseline_results['best_vals'][0]:.10e}")
print(f"Melhor rodada baseline -> f(x) = {baseline_results['min_final']:.10e}")

best_idx_opt = np.argmin(optimized_results['best_vals'])
print(f"\nOtimizada      -> melhor x* = [{optimized_results['best_vars'][best_idx_opt][0]:.10f}, {optimized_results['best_vars'][best_idx_opt][1]:.10f}], f(x*) = {optimized_results['best_vals'][best_idx_opt]:.10e}")
print(f"Melhor rodada otimizada -> f(x) = {optimized_results['min_final']:.10e}")