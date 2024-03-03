from ga import GA
from data import scenes


ga = GA()
ga.set_scene_list(scenes)

i1 = [1] * 8

print("8 长度的染色体的启发值为", ga.fitness_by_scene8(i1))