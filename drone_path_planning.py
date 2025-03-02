import matplotlib.pyplot as plt
import numpy as np
import json
import heapq
import time
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace

# Configuración inicial - SIMPLIFICADA
MAP_WIDTH = 6
MAP_HEIGHT = 4
RESOLUTION = 0.1  # Resolución más gruesa para mejor rendimiento
DRONE_SIZE = 0.2
SAFE_DISTANCE = 0.4  # Distancia de seguridad entre drones
MAX_ITERATIONS = 8000  # Reducido para mejor rendimiento
OBSTACLE_CLEARANCE = 0.08  # Reducido significativamente
DETECTION_RADIUS = 0.2  # Aumentado para mejor detección de objetivos
SIMULATION_SPEED = 0.08  # Velocidad de simulación

# Posiciones de los drones
drone_positions = [(-1.5, -2), (-2, 0)]
drone_colors = ['green', 'red']

# Obstáculos
obstacles = [
    [(-0.218, 0.226), (-0.746, -0.079), (-0.442, -0.607), (0.086, -0.302)],
    [(0.442, 0.607), (-0.086, 0.302), (0.218, -0.226), (0.746, 0.079)],
    [(-0.6952, 1.3048), (-1.3048, 1.3048), (-1.3048, 0.6952), (-0.6952, 0.6952)],
    [(1.3048, -0.6952), (0.6952, -0.6952), (0.6952, -1.3048), (1.3048, -1.3048)],
    [(0.8733, 1.4355), (0.2845, 1.5933), (0.1267, 1.0045), (0.7155, 0.8467)],
    [(-0.5689, -1), (-1, -0.5689), (-1.431, -1), (-1, -1.431)]
]

# Posiciones objetivo originales
original_target_positions = [
    (-0.867, -0.357), (-0.277, 0.550), (0.286, -0.497), (-1.017, 1.527), 
    (0.673, 0.629), (-1.378, -1.369), (1.545, -0.999)
]

# Clase para representar un punto en el espacio
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):
        return hash((round(self.x * 1000), round(self.y * 1000)))
    
    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f})"

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calcula la distancia de un punto a un segmento de línea"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return np.sqrt(A * A + B * B)
    
    param = max(0, min(1, dot / len_sq))
    
    xx = x1 + param * C
    yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    
    return np.sqrt(dx * dx + dy * dy)

# Clase para el agente Drone - MODIFICADA para asegurar que el drone 1 visita todos los puntos
class DroneAgent(Agent):
    def __init__(self, unique_id, model, position, color, targets, target_assignment=None):
        super().__init__(unique_id, model)
        self.position = position
        self.color = color
        
        # Targets como lista simple para iterarlos uno a uno
        self.targets = list(targets)
        
        # Asignación específica de targets para cada drone (nuevo)
        self.target_assignment = target_assignment
        
        # Estado básico
        self.path = []
        self.current_path = []
        self.completed_path = []
        self.target_idx = 0
        self.returning_home = False
        self.is_active = False
        self.completed = False
        self.original_position = position
        
        # Conjunto para rastrear objetivos completados
        self.visited_targets = set()
        
        # Variables para detectar estancamiento
        self.stuck_counter = 0
        self.stuck_detector_time = time.time()
        self.last_positions = []
        
        # Prioridad del drone
        self.priority = 1 if unique_id == 1 else 2
        
        # Tiempos de pausa
        self.waiting = False
        self.wait_until = 0
        
        # Intentos de alcanzar un objetivo (nuevo)
        self.target_attempts = {}
    
    def step(self):
        if not self.is_active or self.completed:
            return
        
        # Comprobar si estamos esperando
        current_time = time.time()
        if self.waiting and current_time < self.wait_until:
            return
        self.waiting = False
        
        # Comprobar si estamos atascados
        self.check_if_stuck()
        
        # Comprobar si estamos cerca de algún objetivo
        current_pos = Point(self.position[0], self.position[1])
        for i, target in enumerate(self.targets):
            target_point = Point(target[0], target[1])
            if current_pos.distance(target_point) < DETECTION_RADIUS:
                self.visited_targets.add(i)
                print(f"Drone {self.unique_id}: Objetivo {i} alcanzado!")
        
        # Si no tenemos un camino, planificar uno
        if not self.current_path:
            if self.returning_home:
                # Verificar si hemos llegado a casa
                home_point = Point(self.original_position[0], self.original_position[1])
                if current_pos.distance(home_point) < DETECTION_RADIUS:
                    self.completed = True
                    print(f"Drone {self.unique_id}: De vuelta en casa. Misión completada!")
                    return
                
                # Planificar camino a casa
                self.plan_simple_path(self.original_position)
            else:
                # Determinar próximo objetivo no visitado
                next_target_idx = None
                
                # Si hay una asignación específica, usarla primero
                if self.target_assignment:
                    for i in self.target_assignment:
                        if i not in self.visited_targets:
                            next_target_idx = i
                            break
                
                # Si no hay asignación o ya completó sus objetivos asignados, buscar cualquier objetivo no visitado
                if next_target_idx is None:
                    for i in range(len(self.targets)):
                        if i not in self.visited_targets:
                            next_target_idx = i
                            break
                
                # Si hemos visitado todos los objetivos, volver a casa
                if next_target_idx is None:
                    self.returning_home = True
                    print(f"Drone {self.unique_id}: Todos los objetivos visitados, volviendo a casa.")
                    self.plan_simple_path(self.original_position)
                else:
                    # Intentar planificar camino hacia el siguiente objetivo
                    target = self.targets[next_target_idx]
                    
                    # Incrementar contador de intentos
                    self.target_attempts[next_target_idx] = self.target_attempts.get(next_target_idx, 0) + 1
                    
                    # Si hemos intentado demasiadas veces, aumentar el radio de detección
                    temp_detection_radius = DETECTION_RADIUS
                    if self.target_attempts.get(next_target_idx, 0) > 5:
                        temp_detection_radius = DETECTION_RADIUS * 1.5
                        print(f"Drone {self.unique_id}: Aumentando radio de detección para objetivo {next_target_idx}")
                    
                    success = self.plan_path_with_retry(target, next_target_idx, temp_detection_radius)
                    
                    if not success and self.target_attempts.get(next_target_idx, 0) > 10:
                        # Si falla demasiadas veces, marcar como visitado y saltar al siguiente
                        # Solo para el drone 2, para que el drone 1 siga intentando
                        if self.unique_id != 1 or next_target_idx not in [1]:  # Aquí se especifica que el drone 1 no debe saltar el punto 1
                            print(f"Drone {self.unique_id}: No se pudo alcanzar objetivo {next_target_idx} después de múltiples intentos, saltando.")
                            self.visited_targets.add(next_target_idx)
        
        # Comprobar colisión con otro drone antes de moverse
        if self.current_path:
            next_pos = self.current_path[0]
            collision_detected = False
            
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id and agent.is_active:
                    agent_pos = Point(agent.position[0], agent.position[1])
                    if next_pos.distance(agent_pos) < SAFE_DISTANCE:
                        collision_detected = True
                        # El de menor prioridad espera
                        if self.priority > agent.priority:
                            self.waiting = True
                            self.wait_until = time.time() + 0.5
                            return
                        else:
                            # El de mayor prioridad espera brevemente 
                            self.waiting = True
                            self.wait_until = time.time() + 0.1
                            return
            
            # Si no hay colisión, moverse
            if not collision_detected:
                self.position = (next_pos.x, next_pos.y)
                self.completed_path.append((next_pos.x, next_pos.y))
                self.current_path.pop(0)
    
    def plan_path_with_retry(self, target, target_idx, detection_radius=DETECTION_RADIUS):
        """Intenta planificar una ruta con varios métodos si es necesario"""
        # Primer intento: ruta normal
        success = self.plan_simple_path(target)
        if success:
            return True
        
        # Segundo intento: enfoque adaptativo
        print(f"Drone {self.unique_id}: Recalculando ruta adaptativa para objetivo {target_idx}")
        # Obtener varios puntos alrededor del objetivo
        target_point = Point(target[0], target[1])
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        distance = 0.3  # Distancia desde el objetivo
        
        for angle in angles:
            rad = np.radians(angle)
            aux_x = target_point.x + distance * np.cos(rad)
            aux_y = target_point.y + distance * np.sin(rad)
            aux_point = (aux_x, aux_y)
            
            # Si podemos planificar una ruta a este punto auxiliar
            if not self.is_point_in_obstacle(Point(aux_x, aux_y)) and self.plan_simple_path(aux_point):
                return True
        
        # Si todos los intentos fallan
        return False
    
    def check_if_stuck(self):
        """Método simplificado para detectar si el drone está atascado"""
        current_time = time.time()
        
        # Verificar cada segundo
        if current_time - self.stuck_detector_time > 1.0:
            self.stuck_detector_time = current_time
            
            # Añadir posición actual
            current_pos = (round(self.position[0], 3), round(self.position[1], 3))
            self.last_positions.append(current_pos)
            
            # Mantener solo últimas 5 posiciones
            if len(self.last_positions) > 5:
                self.last_positions.pop(0)
            
            # Verificar si todas son similares
            if len(self.last_positions) >= 5:
                all_similar = True
                for pos in self.last_positions[1:]:
                    if abs(pos[0] - self.last_positions[0][0]) > 0.02 or abs(pos[1] - self.last_positions[0][1]) > 0.02:
                        all_similar = False
                        break
                
                if all_similar:
                    self.stuck_counter += 1
                    if self.stuck_counter >= 2:
                        print(f"Drone {self.unique_id} atascado. Recalculando ruta.")
                        self.current_path = []
                        self.stuck_counter = 0
                else:
                    self.stuck_counter = 0
    
    def plan_simple_path(self, target):
        """Método de planificación simplificado que combina A* básico con camino directo"""
        start = Point(self.position[0], self.position[1])
        goal = Point(target[0], target[1])
        
        # Si la distancia es pequeña, crear camino directo
        if start.distance(goal) < 0.4:
            direct_path = []
            num_segments = 20
            
            for i in range(num_segments + 1):
                t = i / num_segments
                x = start.x + t * (goal.x - start.x)
                y = start.y + t * (goal.y - start.y)
                point = Point(x, y)
                
                # Verificar colisión solo con obstáculos para rutas cortas
                if self.is_point_in_obstacle(point):
                    break
                
                direct_path.append(point)
            
            if len(direct_path) > num_segments / 2:
                self.current_path = direct_path
                return True
        
        # Usar A* simplificado
        path = self.simplified_a_star(start, goal)
        
        if path:
            self.current_path = path
            return True
        else:
            return False
    
    def is_point_in_obstacle(self, point):
        """Verifica si un punto está dentro de un obstáculo"""
        # Verificar límites del mapa
        if (point.x < -MAP_WIDTH/2 or point.x > MAP_WIDTH/2 or 
            point.y < -MAP_HEIGHT/2 or point.y > MAP_HEIGHT/2):
            return True
        
        # Verificar colisión con obstáculos
        for obstacle in obstacles:
            # Dentro del polígono
            if self.point_in_polygon(point.x, point.y, obstacle):
                return True
            
            # Muy cerca del borde
            for i in range(len(obstacle)):
                p1 = obstacle[i]
                p2 = obstacle[(i + 1) % len(obstacle)]
                dist = point_to_segment_distance(point.x, point.y, p1[0], p1[1], p2[0], p2[1])
                if dist < OBSTACLE_CLEARANCE:
                    return True
        
        return False
    
    def point_in_polygon(self, x, y, polygon):
        """Determina si un punto está dentro de un polígono utilizando el algoritmo de ray casting"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def simplified_a_star(self, start, goal):
        """Implementación simplificada de A* para rutas robustas"""
        # Direcciones de movimiento (8 direcciones)
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # Cardinales
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonales
        ]
        
        # Normalizar direcciones
        normalized_directions = []
        for dx, dy in directions:
            length = np.sqrt(dx**2 + dy**2)
            normalized_directions.append((dx/length * RESOLUTION, dy/length * RESOLUTION))
        
        # Inicializar A*
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {(round(start.x, 2), round(start.y, 2)): 0}
        f_score = {(round(start.x, 2), round(start.y, 2)): start.distance(goal)}
        counter = 0
        closed_set = set()
        iterations = 0
        
        # Iniciar búsqueda
        start_time = time.time()
        max_search_time = 0.5  # Máximo medio segundo por búsqueda
        
        while open_set and iterations < MAX_ITERATIONS:
            # Verificar tiempo límite
            if time.time() - start_time > max_search_time:
                print(f"Tiempo de búsqueda excedido para drone {self.unique_id}")
                break
            
            iterations += 1
            _, _, current = heapq.heappop(open_set)
            current_key = (round(current.x, 2), round(current.y, 2))
            
            # Verificar si hemos llegado al objetivo
            if current.distance(goal) < DETECTION_RADIUS:
                path = []
                while current_key in came_from:
                    path.append(Point(current_key[0], current_key[1]))
                    current_key = came_from[current_key]
                path.append(start)
                path.reverse()
                path.append(goal)
                return self.simple_path_smoothing(path)
            
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            for dx, dy in normalized_directions:
                neighbor = Point(current.x + dx, current.y + dy)
                neighbor_key = (round(neighbor.x, 2), round(neighbor.y, 2))
                
                # Verificar colisión
                if self.is_point_in_obstacle(neighbor):
                    continue
                
                # Calcular puntuación g
                tentative_g_score = g_score[current_key] + current.distance(neighbor)
                
                if neighbor_key in closed_set and tentative_g_score >= g_score.get(neighbor_key, float('inf')):
                    continue
                
                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = current_key
                    g_score[neighbor_key] = tentative_g_score
                    f_score[neighbor_key] = tentative_g_score + 1.2 * neighbor.distance(goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor_key], counter, neighbor))
        
        # Si no encontramos camino, y estamos cerca del objetivo, intentar camino directo
        if start.distance(goal) < 0.8:
            direct_path = [start]
            segments = 10
            for i in range(1, segments + 1):
                t = i / segments
                x = start.x + t * (goal.x - start.x)
                y = start.y + t * (goal.y - start.y)
                direct_path.append(Point(x, y))
            return direct_path
        
        # No se encontró camino
        return []
    
    def simple_path_smoothing(self, path):
        """Versión muy simplificada del suavizado de ruta"""
        if len(path) <= 2:
            return path
        
        # Insertar puntos adicionales en segmentos largos
        smooth_path = [path[0]]
        
        for i in range(1, len(path)):
            prev = path[i-1]
            current = path[i]
            
            # Si la distancia es grande, insertar puntos intermedios
            distance = prev.distance(current)
            if distance > RESOLUTION * 2:
                num_points = int(distance / RESOLUTION)
                for j in range(1, num_points):
                    t = j / num_points
                    x = prev.x + t * (current.x - prev.x)
                    y = prev.y + t * (current.y - prev.y)
                    smooth_path.append(Point(x, y))
            
            smooth_path.append(current)
        
        return smooth_path

# Modelo de simulación - MODIFICADO
class DroneModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(MAP_WIDTH, MAP_HEIGHT, True)
        
        # Asignar objetivos específicos para cada drone
        # Drone 1 (verde) debe visitar todos los objetivos, pero damos prioridad a algunos
        drone1_targets = [0, 2, 3, 4, 5, 6]
        # Drone 2 (rojo) se enfocará solo en el punto 1 y luego ayudará con otros
        drone2_targets = [1, 6]
        
        # Crear los drones con asignaciones específicas
        self.drone1 = DroneAgent(1, self, drone_positions[0], drone_colors[0], 
                                original_target_positions, target_assignment=drone1_targets)
        self.drone2 = DroneAgent(2, self, drone_positions[1], drone_colors[1], 
                                original_target_positions, target_assignment=drone2_targets)
        
        # Activar solo el primer drone al principio
        self.drone1.is_active = True
        
        # Añadir los drones al scheduler
        self.schedule.add(self.drone1)
        self.schedule.add(self.drone2)
        
        # Activación del segundo drone
        self.drone2_activation_time = None
        
        # Tiempo de inicio
        self.start_time = time.time()
    
    def step(self):
        # Ejecutar un paso en los agentes
        self.schedule.step()
        
        # Activar drone 2 cuando drone 1 haya visitado al menos 2 objetivos
        if not self.drone2.is_active and len(self.drone1.visited_targets) >= 2:
            if self.drone2_activation_time is None:
                self.drone2_activation_time = time.time()
                print("Drone 1 ha visitado suficientes objetivos. Drone 2 se activará pronto.")
            
            # Esperar 2 segundos antes de activar
            if time.time() - self.drone2_activation_time > 2.0:
                self.drone2.is_active = True
                print("Activando drone 2")
        
        # Comprobar si los drones están muy cerca y resolver conflicto
        if self.drone1.is_active and self.drone2.is_active:
            d1_pos = Point(self.drone1.position[0], self.drone1.position[1])
            d2_pos = Point(self.drone2.position[0], self.drone2.position[1])
            
            if d1_pos.distance(d2_pos) < SAFE_DISTANCE * 0.7:
                # Resolver conflicto deteniendo temporalmente a uno de los drones
                if self.drone1.priority > self.drone2.priority:
                    self.drone1.waiting = True
                    self.drone1.wait_until = time.time() + 1.0
                else:
                    self.drone2.waiting = True
                    self.drone2.wait_until = time.time() + 1.0
        
        # Verificar progreso de los drones para evitar que se queden atascados en la misión global
        if self.drone1.is_active and not self.drone1.completed:
            # Si el drone 1 lleva mucho tiempo sin visitar nuevos objetivos, revisar sus asignaciones
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 60 and len(self.drone1.visited_targets) < 3:
                print("Drone 1 está progresando lentamente. Ajustando comportamiento.")
                # Reducir temporalmente la prioridad del drone 1 para evitar bloqueos
                self.drone1.priority = 2
                self.drone2.priority = 1
        
        # Verificar si se han completado ambos drones
        drone1_done = self.drone1.completed
        drone2_done = self.drone2.completed if self.drone2.is_active else True
        
        # Verificar si el drone 1 ha visitado todos los puntos
        # Si el drone 2 visitó el punto 1 pero el drone 1 no, hacer que el drone 1
        # lo visite antes de terminar
        if drone1_done and 1 not in self.drone1.visited_targets and 1 in self.drone2.visited_targets:
            print("El drone 1 debe visitar el punto 1 antes de completar la misión.")
            self.drone1.completed = False
            self.drone1.returning_home = False
            drone1_done = False
        
        # Verificar tiempo máximo (5 minutos)
        timeout = (time.time() - self.start_time) > 300
        
        if (drone1_done and drone2_done) or timeout:
            return True
        
        return False

# Función principal
def run_simulation():
    # Crear modelo
    model = DroneModel()
    
    # Configurar visualización
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("X (metros)")
    ax.set_ylabel("Y (metros)")
    ax.set_title("Simulación de Ruta del Dron")
    ax.grid(True)
    
    # Dibujar obstáculos
    for obs in obstacles:
        x_coords, y_coords = zip(*obs)
        ax.fill(x_coords, y_coords, color='blue', alpha=0.6)
    
    # Dibujar posiciones objetivo con etiquetas
    for i, (x, y) in enumerate(original_target_positions):
        # Destacar el punto 1 que es el que se debe visitar
        if i == 1:
            ax.plot(x, y, 'ro', markersize=8)
            ax.text(x + 0.05, y + 0.05, str(i), fontsize=10, fontweight='bold')
        else:
            ax.plot(x, y, 'ro')
            ax.text(x + 0.05, y + 0.05, str(i), fontsize=9)
    
    # Dibujar drones
    drone1_rect = plt.Rectangle(
        (drone_positions[0][0] - DRONE_SIZE/2, drone_positions[0][1] - DRONE_SIZE/2),
        DRONE_SIZE, DRONE_SIZE, color=drone_colors[0]
    )
    drone2_rect = plt.Rectangle(
        (drone_positions[1][0] - DRONE_SIZE/2, drone_positions[1][1] - DRONE_SIZE/2),
        DRONE_SIZE, DRONE_SIZE, color=drone_colors[1]
    )
    ax.add_patch(drone1_rect)
    ax.add_patch(drone2_rect)
    
    # Etiquetar drones
    ax.text(drone_positions[0][0] - 0.1, drone_positions[0][1] - 0.3, "Drone 1", fontsize=9)
    ax.text(drone_positions[1][0] - 0.1, drone_positions[1][1] - 0.3, "Drone 2", fontsize=9)
    
    # Ejecutar la simulación
    all_done = False
    drone_paths = {"drone_1": [], "drone_2": []}
    
    start_time = time.time()
    step_count = 0
    
    while not all_done:
        step_count += 1
        
        # Comprobar tiempo máximo (10 minutos)
        if time.time() - start_time > 600:
            print("¡Tiempo máximo de ejecución excedido!")
            break
            
        # Actualizar el modelo
        all_done = model.step()
        
        # Actualizar la visualización de los drones
        drone1_rect.set_xy((model.drone1.position[0] - DRONE_SIZE/2, model.drone1.position[1] - DRONE_SIZE/2))
        drone2_rect.set_xy((model.drone2.position[0] - DRONE_SIZE/2, model.drone2.position[1] - DRONE_SIZE/2))
        
        # Almacenar las posiciones actuales
        if model.drone1.is_active and len(model.drone1.completed_path) > 0:
            last_pos = model.drone1.completed_path[-1]
            if last_pos not in drone_paths["drone_1"]:
                drone_paths["drone_1"].append(last_pos)
                ax.plot(last_pos[0], last_pos[1], 'go', alpha=0.6)
        
        if model.drone2.is_active and len(model.drone2.completed_path) > 0:
            last_pos = model.drone2.completed_path[-1]
            if last_pos not in drone_paths["drone_2"]:
                drone_paths["drone_2"].append(last_pos)
                ax.plot(last_pos[0], last_pos[1], 'bo', alpha=0.6)
        
        # Actualizar la visualización
        plt.pause(SIMULATION_SPEED)
        
        # Mostrar progreso cada 100 pasos
        if step_count % 100 == 0:
            print(f"Paso {step_count}:")
            print(f"  Drone 1: {len(model.drone1.visited_targets)}/{len(original_target_positions)} objetivos")
            print(f"  Drone 2: {len(model.drone2.visited_targets)}/{len(original_target_positions)} objetivos")
    
    # Mostrar resumen final
    print("\nResumen final:")
    print(f"Drone 1 visitó {len(model.drone1.visited_targets)}/{len(original_target_positions)} objetivos")
    print(f"Drone 2 visitó {len(model.drone2.visited_targets)}/{len(original_target_positions)} objetivos")
    
    # Guardar las rutas
    with open("TargetPositions.txt", "w") as file:
        json.dump(drone_paths, file, indent=4)
    print("Ruta guardada en TargetPositions.txt")
    
    plt.show()

if __name__ == "__main__":
    print("Iniciando simulación con enfoque en visitar todos los puntos")
    print(f"El drone 1 (verde) visitará todos los puntos")
    run_simulation()