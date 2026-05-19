# -*- coding: utf-8 -*-
"""
SCRIPTS PRONTOS PARA EXECUTAR EXPERIMENTOS COMPARATIVOS

Cada bloco abaixo executa um experimento variando um parâmetro do GA.
Descomente o bloco desejado e execute para gerar dados e gráficos.

INSTRUÇÕES:
1. Descomente UMA SEÇÃO DE CADA VEZ
2. Execute o script
3. Analise os gráficos e a tabela de resultados
4. Copie os dados da tabela para seu relatório
5. Use o gráfico para discutir os resultados
"""

import sys
import importlib.util

sys.path.insert(0, '/home/cameratothiago/programs/ENG4456-T2')

# Carregar módulo com espaço no nome usando importlib
spec = importlib.util.spec_from_file_location("trabalho_ga", "/home/cameratothiago/programs/ENG4456-T2/Trabalho 2 GA_2025.py")
trabalho_ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trabalho_ga)

# Importar as funções e classes do módulo
f = trabalho_ga.f
varbound = trabalho_ga.varbound
algorithm_param = trabalho_ga.algorithm_param
ExperimentComparison = trabalho_ga.ExperimentComparison

# Baseline (referência) - MANTER IGUAL PARA TODOS OS TESTES
baseline_config = {
    'max_num_iteration': 300,
    'population_size': 100,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'one_point',
    'max_iteration_without_improv': None,
    'mutation_type': 'uniform_by_center',
    'selection_type': 'roulette'
}

print("="*80)
print("EXPERIMENTOS COMPARATIVOS - ALGORITMO GENÉTICO PARA RASTRIGIN")
print("="*80)
"""

#############################################################################################
# EXPERIMENTO 1: EFEITO DO population_size
#############################################################################################
print("\n[1] EXPERIMENTO: population_size")
print("Testando: [50, 100, 200]")
print("Baseline: 100")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('population_size', [50, 100, 200], verbose=True)
comparator.print_report('population_size')
comparator.plot_comparison('population_size')
comparator.plot_all_convergences('population_size')  # Novo gráfico com cores diferentes
"""

"""
#############################################################################################
# EXPERIMENTO 2: EFEITO DO mutation_probability
#############################################################################################
print("\n[2] EXPERIMENTO: mutation_probability")
print("Testando: [0.01, 0.1, 0.3]")
print("Baseline: 0.1")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('mutation_probability', [0.01, 0.1, 0.3], verbose=True)
comparator.print_report('mutation_probability')
comparator.plot_comparison('mutation_probability')
comparator.plot_all_convergences('mutation_probability')  # Novo gráfico com cores diferentes


#############################################################################################
# EXPERIMENTO 3: EFEITO DO elit_ratio
#############################################################################################
print("\n[3] EXPERIMENTO: elit_ratio")
print("Testando: [0.0, 0.05, 0.1]")
print("Baseline: 0.01")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('elit_ratio', [0.0, 0.01, 0.1], verbose=True)
comparator.print_report('elit_ratio')
comparator.plot_comparison('elit_ratio')
comparator.plot_all_convergences('elit_ratio')

#############################################################################################
# EXPERIMENTO 4: EFEITO DO crossover_probability
#############################################################################################
print("\n[4] EXPERIMENTO: crossover_probability")
print("Testando: [0.3, 0.5, 0.8]")
print("Baseline: 0.5")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('crossover_probability', [0.3, 0.5, 0.8], verbose=True)
comparator.print_report('crossover_probability')
comparator.plot_comparison('crossover_probability')
comparator.plot_all_convergences('crossover_probability')  # Novo gráfico com cores diferentes


#############################################################################################
# EXPERIMENTO 5: EFEITO DO parents_portion
#############################################################################################
print("\n[5] EXPERIMENTO: parents_portion")
print("Testando: [0.1, 0.3, 0.6]")
print("Baseline: 0.3")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('parents_portion', [0.1, 0.3, 0.6], verbose=True)
comparator.print_report('parents_portion')
comparator.plot_comparison('parents_portion')
comparator.plot_all_convergences('parents_portion')  # Novo gráfico com cores diferentes

#############################################################################################
# EXPERIMENTO 6: EFEITO DO crossover_type
#############################################################################################
print("\n[6] EXPERIMENTO: crossover_type")
print("Testando: ['uniform', 'one_point', 'two_point']")
print("Baseline: 'one_point'")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('crossover_type', ['uniform', 'one_point', 'two_point'], verbose=True)
comparator.print_report('crossover_type')
comparator.plot_comparison('crossover_type')
comparator.plot_all_convergences('crossover_type')  # Novo gráfico com cores diferentes


#############################################################################################
# EXPERIMENTO 7: EFEITO DO selection_type
#############################################################################################
print("\n[7] EXPERIMENTO: selection_type")
print("Testando: ['tournament', 'roulette', 'fully_random']")
print("Baseline: 'tournament'")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('selection_type', ['tournament', 'roulette', 'fully_random'], verbose=True)
comparator.print_report('selection_type')
comparator.plot_comparison('selection_type')
comparator.plot_all_convergences('selection_type')  # Novo gráfico com cores diferentes


#############################################################################################
# EXPERIMENTO 8: EFEITO DO max_num_iteration
#############################################################################################
print("\n[8] EXPERIMENTO: max_num_iteration")
print("Testando: [50, 300, 1000]")
print("Baseline: 300")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('max_num_iteration', [50, 300, 1000], verbose=True)
comparator.print_report('max_num_iteration')
comparator.plot_comparison('max_num_iteration')
comparator.plot_all_convergences('max_num_iteration')  # Novo gráfico com cores diferentes
"""
#############################################################################################
# EXPERIMENTO 9: EFEITO DO max_iteration_without_improv
#############################################################################################
print("\n[9] EXPERIMENTO: max_iteration_without_improv")
print("Testando: [50, None, 100]")
print("Baseline: None")

comparator = ExperimentComparison(
    objective_function=f,
    variable_boundaries=varbound,
    baseline_params=baseline_config,
    num_experiments=20
)
comparator.run_experiment_set('max_iteration_without_improv', [50, None, 100], verbose=True)
comparator.print_report('max_iteration_without_improv')
comparator.plot_comparison('max_iteration_without_improv')
comparator.plot_all_convergences('max_iteration_without_improv')  # Novo gráfico com cores diferentes
