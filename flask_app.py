import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

POP_SIZE = 50
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1

def run_optimization(room_length, room_width, room_height, num_acs,
                     user_ac_positions, window_positions, door_positions):

    def is_near_structure(ac, structures, min_distance):
        return any(np.linalg.norm(np.array(ac) - np.array(s)) < min_distance for s in structures)

    def wall_penalty(ac):
        return min(ac[0], room_length - ac[0], ac[1], room_width - ac[1])

    def energy_efficiency_score(layout):
        score = 0
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                score += np.linalg.norm(np.array(layout[i]) - np.array(layout[j]))
        pairs = len(layout) * (len(layout) - 1) / 2
        return score / (pairs + 1)

    def fitness(layout):
        score = 100
        for ac in layout:
            if is_near_structure(ac, window_positions, 3):
                score -= 10
            if is_near_structure(ac, door_positions, 3):
                score -= 10
            if wall_penalty(ac) < 2:
                score -= 5

        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                dist = np.linalg.norm(np.array(layout[i]) - np.array(layout[j]))
                if dist < 6:
                    score -= 10
                elif dist < 10:
                    score -= 5

        score -= energy_efficiency_score(layout) * 1.5
        return max(score, 0)

    def generate_layout():
        layout = []
        z_min = max(5, room_height - 3)
        for _ in range(num_acs):
            wall = random.choice(["left", "right", "front", "back"])
            if wall == "left":
                x = 0
                y = random.randint(2, room_width - 3)
            elif wall == "right":
                x = room_length
                y = random.randint(2, room_width - 3)
            elif wall == "front":
                y = 0
                x = random.randint(2, room_length - 3)
            else:
                y = room_width
                x = random.randint(2, room_length - 3)
            z = random.randint(z_min, room_height - 1)
            layout.append((x, y, z))
        return layout

    population = [generate_layout() for _ in range(POP_SIZE)]

    for gen in range(NUM_GENERATIONS):
        population.sort(key=fitness, reverse=True)
        parents = population[:10]
        next_gen = []
        for _ in range(POP_SIZE):
            p1, p2 = random.sample(parents, 2)
            if num_acs > 1:
                idx = random.randint(1, num_acs - 1)
                child = p1[:idx] + p2[idx:]
            else:
                child = p1.copy()
            if random.random() < MUTATION_RATE:
                idx = random.randint(0, num_acs - 1)
                layout = generate_layout()
                child[idx] = layout[idx]
            next_gen.append(child)
        population = next_gen

    best_layout = max(population, key=fitness)

    initial_energy_efficiency = energy_efficiency_score(user_ac_positions)
    optimized_energy_efficiency = energy_efficiency_score(best_layout)
    initial_fitness = fitness(user_ac_positions)
    optimized_fitness = fitness(best_layout)

    return (best_layout, initial_energy_efficiency, initial_fitness,
            optimized_energy_efficiency, optimized_fitness)

def draw_room(ax, room_length, room_width, room_height, room_offset_x=0):
    walls = [
        [[0, 0, 0], [room_length, 0, 0], [room_length, 0, room_height], [0, 0, room_height]],
        [[0, room_width, 0], [room_length, room_width, 0], [room_length, room_width, room_height], [0, room_width, room_height]],
        [[0, 0, 0], [0, room_width, 0], [0, room_width, room_height], [0, 0, room_height]],
        [[room_length, 0, 0], [room_length, room_width, 0], [room_length, room_width, room_height], [room_length, 0, room_height]],
        [[0, 0, room_height], [room_length, 0, room_height], [room_length, room_width, room_height], [0, room_width, room_height]],
    ]
    offset_walls = [[[x + room_offset_x, y, z] for x, y, z in wall] for wall in walls]
    ax.add_collection3d(Poly3DCollection(offset_walls, facecolors='whitesmoke', linewidths=1, edgecolors='gray', alpha=0.1))

def draw_flat_panel(ax, x, y, z, w, h, wall, color, room_offset_x=0, label=None):
    if wall == 'left' or wall == 'right':
        verts = [[(x + room_offset_x, y - w/2, z - h/2), (x + room_offset_x, y + w/2, z - h/2),
                  (x + room_offset_x, y + w/2, z + h/2), (x + room_offset_x, y - w/2, z + h/2)]]
    elif wall == 'front' or wall == 'back':
        verts = [[(x - w/2 + room_offset_x, y, z - h/2), (x + w/2 + room_offset_x, y, z - h/2),
                  (x + w/2 + room_offset_x, y, z + h/2), (x - w/2 + room_offset_x, y, z + h/2)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, edgecolors='k', linewidths=1))
    if label:
        ax.text(x + room_offset_x, y, z + h/2 + 0.5, label, color='black', fontsize=8)

def detect_wall(x, y, room_length, room_width, tolerance=0.01):
    if abs(x - 0) < tolerance:
        return 'left'
    elif abs(x - room_length) < tolerance:
        return 'right'
    elif abs(y - 0) < tolerance:
        return 'front'
    elif abs(y - room_width) < tolerance:
        return 'back'
    else:
        return 'unknown'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Parse inputs
        room_length = int(request.form['room_length'])
        room_width = int(request.form['room_width'])
        room_height = int(request.form['room_height'])
        num_acs = int(request.form['num_acs'])

        # Parse user AC positions
        user_ac_positions = []
        for i in range(num_acs):
            x = int(request.form.get(f'ac_x_{i}'))
            y = int(request.form.get(f'ac_y_{i}'))
            z = int(request.form.get(f'ac_z_{i}'))
            user_ac_positions.append((x, y, z))

        # Parse windows
        window_count = int(request.form['window_count'])
        window_positions = []
        for i in range(window_count):
            x = int(request.form.get(f'window_x_{i}'))
            y = int(request.form.get(f'window_y_{i}'))
            z = int(request.form.get(f'window_z_{i}'))
            x = min(max(x, 0), room_length) if x in [0, room_length] else x
            y = min(max(y, 0), room_width) if y in [0, room_width] else y
            z = min(max(z, 4), room_height - 1)
            window_positions.append((x, y, z))

        # Parse doors
        door_count = int(request.form['door_count'])
        door_positions = []
        for i in range(door_count):
            x = int(request.form.get(f'door_x_{i}'))
            y = int(request.form.get(f'door_y_{i}'))
            z = int(request.form.get(f'door_z_{i}'))
            x = min(max(x, 0), room_length) if x in [0, room_length] else x
            y = min(max(y, 0), room_width) if y in [0, room_width] else y
            z = 0
            door_positions.append((x, y, z))

        # Run optimization
        (best_layout, initial_efficiency, initial_fitness,
         optimized_efficiency, optimized_fitness) = run_optimization(
             room_length, room_width, room_height, num_acs,
             user_ac_positions, window_positions, door_positions)

        # Debug prints
        print("User AC positions:", user_ac_positions)
        print("Optimized AC positions (best layout):", best_layout)

        # Plotting (optional, if you want to save/show the image)
        fig = plt.figure(figsize=(14, 7))

        # Left subplot: user layout
        ax1 = fig.add_subplot(121, projection='3d')
        draw_room(ax1, room_length, room_width, room_height)
        for idx, (x, y, z) in enumerate(user_ac_positions):
            wall = detect_wall(x, y, room_length, room_width)
            draw_flat_panel(ax1, x, y, z, 2.7, 1.2, wall, 'blue', label=f"AC {idx+1}")
        for x, y, z in window_positions:
            wall = detect_wall(x, y, room_length, room_width)
            draw_flat_panel(ax1, x, y, z, 3, 3, wall, 'cyan')
        for x, y, z in door_positions:
            wall = detect_wall(x, y, room_length, room_width)
            draw_flat_panel(ax1, x, y, z, 4, 6, wall, 'brown')
        ax1.set_title("User AC Layout")
        ax1.set_xlim([-1, room_length + 1])
        ax1.set_ylim([-1, room_width + 1])
        ax1.set_zlim([0, room_height + 1])
        ax1.set_xlabel("Length (m)")
        ax1.set_ylabel("Width (m)")
        ax1.set_zlabel("Height (m)")

        # Right subplot: optimized layout
        ax2 = fig.add_subplot(122, projection='3d')
        offset_x = room_length + 10
        draw_room(ax2, room_length, room_width, room_height, room_offset_x=offset_x)
        for idx, (x, y, z) in enumerate(best_layout):
            wall = detect_wall(x, y, room_length, room_width)
            if wall == 'unknown':
                print(f"Warning: AC {idx+1} at ({x:.2f}, {y:.2f}, {z:.2f}) is not on any wall!")
            draw_flat_panel(ax2, x, y, z, 2.7, 1.2, wall, 'green', room_offset_x=offset_x, label=f"AC {idx+1}")
        for x, y, z in window_positions:
            wall = detect_wall(x, y, room_length, room_width)
            draw_flat_panel(ax2, x, y, z, 3, 3, wall, 'cyan', room_offset_x=offset_x)
        for x, y, z in door_positions:
            wall = detect_wall(x, y, room_length, room_width)
            draw_flat_panel(ax2, x, y, z, 4, 6, wall, 'brown', room_offset_x=offset_x)
        ax2.set_title("Optimized AC Layout")
        ax2.set_xlim([offset_x - 1, offset_x + room_length + 1])
        ax2.set_ylim([-1, room_width + 1])
        ax2.set_zlim([0, room_height + 1])
        ax2.set_xlabel("Length (m)")
        ax2.set_ylabel("Width (m)")
        ax2.set_zlabel("Height (m)")

        plt.tight_layout()
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "room_layout.png")
        plt.savefig(output_path)
        plt.close()

        # Return the result template with the metrics
        return render_template('result.html',
                               initial_efficiency=round(initial_efficiency, 3),
                               optimized_efficiency=round(optimized_efficiency, 3),
                               initial_fitness=round(initial_fitness, 3),
                               optimized_fitness=round(optimized_fitness, 3))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
