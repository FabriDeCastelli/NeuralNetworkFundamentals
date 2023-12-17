import os

monk_benchmark_path = "/data_for_testing/monk/monk-{}"
ml_cup_path = "/data_for_testing/cup"

project_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
monk_benchmark_path = project_folder_path + monk_benchmark_path
ml_cup_path = project_folder_path + ml_cup_path
