from typing import List, Tuple, Dict
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from math import sqrt

# 1. CREACIÓN DE NODOS
def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Crea un nodo que representa un punto en la cuadrícula.
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }


# 2. CÁLCULO DE HEURÍSTICA
def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calcula la distancia heurística entre dos puntos usando la distancia Euclidiana.
    """
    return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)


# 3. OBTENCIÓN DE VECINOS VÁLIDOS (SIN DIAGONALES)
def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Obtiene los vecinos válidos de una celda en la cuadrícula (sin diagonales).
    """
    x, y = position
    rows, cols = grid.shape

    # Movimientos permitidos (arriba, abajo, izquierda, derecha)
    possible_moves = [
        (x+1, y),  # Abajo
        (x-1, y),  # Arriba
        (x, y+1),  # Derecha
        (x, y-1)   # Izquierda
    ]

    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0
    ]


# 4. RECONSTRUCCIÓN DEL CAMINO
def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruye el camino desde el nodo objetivo hasta el inicio.
    """
    path = []
    current = goal_node
    while current:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]  # Se invierte para obtener el camino en orden correcto


# 5. IMPLEMENTACIÓN DEL ALGORITMO A*
def find_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Encuentra el camino óptimo desde el inicio hasta la meta usando A*.
    """
    start_node = create_node(position=start, g=0, h=calculate_heuristic(start, goal))
    open_list = [(start_node['f'], id(start_node), start)]  # Añadir id para desempate
    open_dict = {start: start_node}         
    closed_set = set()                      

    while open_list:
        _, _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # Si ya procesamos esta posición, continuamos
        if current_pos in closed_set:
            continue
            
        # Si llegamos a la meta, reconstruimos el camino
        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue

            tentative_g = current_node['g'] + 1  # Cada movimiento cuesta 1

            if neighbor_pos not in open_dict or tentative_g < open_dict[neighbor_pos]['g']:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                open_dict[neighbor_pos] = neighbor
                heapq.heappush(open_list, (neighbor['f'], id(neighbor), neighbor_pos))

    return []  # Si no se encuentra camino


# 6. VISUALIZACIÓN DEL CAMINO MEJORADA
def visualize_path_improved(grid: np.ndarray, path: List[Tuple[int, int]], start_pos: Tuple[int, int], 
                            goal_pos: Tuple[int, int], rectangles: List, target_positions: List[Tuple[int, int]]):
    """
    Muestra la cuadrícula con el camino encontrado y obstáculos visualmente atractivos.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Fondo blanco con puntos grises
    ax.set_facecolor('white')
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.plot(j, i, 'o', color='lightgray', markersize=1)
    
    # Dibujar rectángulos (góndolas) en rojo
    for rect in rectangles:
        x, y, width, height, angle = rect
        # Crear un rectángulo
        r = Rectangle((y, x), width, height, angle=angle, facecolor='red', alpha=0.7)
        ax.add_patch(r)
    
    # Dibujar el camino
    if path:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='Camino')
    
    # Marcar inicio y meta
    ax.plot(start_pos[1], start_pos[0], 'o', color='green', markersize=10, label='Inicio')
    
    # Marcar los target positions con círculos azules
    for target in target_positions:
        ax.plot(target[1], target[0], 'o', color='blue', markersize=8, label='Target')
    
    # Configurar gráfico
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)  # Invertir eje y para que 0 esté arriba
    ax.set_aspect('equal')
    ax.set_title("Almacén con Góndolas y Obstáculos")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()


# 7. CREACIÓN DEL MAPA DEL ALMACÉN CORRECTAMENTE
SIZE = 60  # Tamaño de la cuadrícula
grid = np.zeros((SIZE, SIZE))

# Definir rectángulos (góndolas) como (x, y, ancho, alto, ángulo)
rectangles = [
    (10, 5, 10, 10, 45),     # Góndola rotada
    (20, 15, 10, 10, 0),     # Góndola recta 1
    (30, 25, 10, 10, 0),     # Góndola recta 2
    (45, 15, 10, 10, 0),     # Góndola esquina derecha
    (5, 45, 10, 10, 0),      # Góndola esquina izquierda
    (50, 45, 10, 10, 0)       # Nueva góndola
]

# Crear una matriz de ocupación basada en los obstáculos
for i in range(SIZE):
    for j in range(SIZE):
        # Comprobar rectángulos
        for x, y, width, height, angle in rectangles:
            # Simplificado: considerar solo rectángulos sin rotación para ocupación
            if angle == 0:
                if x <= i < x + height and y <= j < y + width:
                    grid[i, j] = 1
            else:
                # Para rectángulos rotados, usamos una aproximación simplificada
                # Crear un cuadrado más grande que cubra el área rotada
                center_x, center_y = x + height/2, y + width/2
                radius = max(width, height) * 0.7  # Un poco más pequeño que la diagonal
                if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                    grid[i, j] = 1

# Definir puntos de inicio y meta
start_pos = (58, 1)           # Entrada del almacén
#goal_pos = (SIZE-1, SIZE-1)  # Zona de salida
# Definir las posiciones objetivo
target_positions_raw = [
    (-0.86717069892473, -0.277318548387096),
    (0.286122311827957, -1.01683467741935),
    (0.673487903225808, -1.37778897849462),
    (1.54506048387097, -0.356552419354838),
    (0.550235215053764, -0.497412634408602),
    (1.52745295698925, 0.629469086021506),
    (-1.36898521505376, -0.999227150537633)
]

# Escalar y desplazar las coordenadas para que encajen en la cuadrícula 60x60
# Asumiendo que las coordenadas originales están en un rango de -2 a 2
SCALE = 14
OFFSET_X = 30  # Desplazamiento en X
OFFSET_Y = 30  # Desplazamiento en Y

target_positions = []
for x, y in target_positions_raw:
    # Escalar y desplazar las coordenadas
    grid_x = int(OFFSET_X - y * SCALE)  # Invertir y para que coincida con la orientación de la cuadrícula
    grid_y = int(OFFSET_Y + x * SCALE)
    
    # Asegurarse de que las coordenadas estén dentro de los límites de la cuadrícula
    grid_x = max(0, min(SIZE - 1, grid_x))
    grid_y = max(0, min(SIZE - 1, grid_y))
    
    target_positions.append((grid_x, grid_y))

# Asegurarse de que el inicio y meta no son obstáculos
grid[start_pos] = 0
#grid[goal_pos] = 0
for target in target_positions:
    grid[target] = 0

# Buscar el camino a través de todos los target positions
full_path = []
current_pos = start_pos

for target in target_positions:
    path = find_path(grid, current_pos, target)
    if path:
        full_path.extend(path)
        current_pos = target  # Actualizar la posición actual al target alcanzado
    else:
        print(f"¡No se encontró un camino a {target}!")
        full_path = []
        break

# Eliminar duplicados manteniendo el orden
full_path_unique = []
seen = set()
for pos in full_path:
    if pos not in seen:
        full_path_unique.append(pos)
        seen.add(pos)

if full_path_unique:
    print(f"¡Camino encontrado con {len(full_path_unique)} pasos!")
    visualize_path_improved(grid, full_path_unique, start_pos, (0,0), rectangles, target_positions)
else:
    print("¡No se encontró un camino a todos los targets!")
    # Visualizar el mapa aunque no haya camino para depurar
    visualize_path_improved(grid, [], start_pos, (0,0), rectangles, target_positions)