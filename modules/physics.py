import numpy as np

class SoleSimulation:
    def __init__(self, length_cm=28, width_cm=10, resolution=1.0):
        self.dx = resolution
        self.nx = int(length_cm / self.dx)
        self.ny = int(width_cm / self.dx)
        self.X, self.Y = np.meshgrid(np.arange(0, self.nx), np.arange(0, self.ny))
        
        self.thickness_map = np.zeros((self.ny, self.nx))
        self.stiffness_map = np.zeros((self.ny, self.nx))
        self.pressure_map = np.zeros((self.ny, self.nx))
        self.wear_map = np.zeros((self.ny, self.nx)) 
        self.current_wear_factor = 0.01
        self.foot_shape = self._generate_foot_shape("Neutral")

    def _generate_foot_shape(self, gait_type="Neutral"):
        heel = np.exp(-((self.X - 5)**2 + (self.Y - self.ny/2)**2) / 20)
        ball = np.exp(-((self.X - 20)**2 + (self.Y - self.ny/2)**2) / 30)
        toes = 0.6 * np.exp(-((self.X - 25)**2 + (self.Y - self.ny/2)**2) / 15)
        
        if gait_type == "Overpronator (Flat Foot)":
            arch_cut = -0.1 * np.exp(-((self.X - 12)**2 + (self.Y - self.ny/1.5)**2) / 20)
            bias = 0.25 * np.exp(-((self.Y - self.ny/3)**2) / 40)
        elif gait_type == "Supinator (High Arch)":
            arch_cut = -0.9 * np.exp(-((self.X - 12)**2 + (self.Y - self.ny/1.5)**2) / 15)
            bias = 0.25 * np.exp(-((self.Y - self.ny*0.8)**2) / 40)
        else:
            arch_cut = -0.5 * np.exp(-((self.X - 12)**2 + (self.Y - self.ny/1.5)**2) / 20)
            bias = 0
            
        shape = (heel + ball + toes + arch_cut + bias)
        return np.clip(shape, 0, 1)

    def update_gait(self, gait_type):
        self.foot_shape = self._generate_foot_shape(gait_type)

    def update_design(self, heel_mm, fore_mm, arch_stiff, modulus, groove_type="None", wear_factor=0.01):
        self.current_wear_factor = wear_factor
        gradient = np.linspace(heel_mm, fore_mm, self.nx)
        self.thickness_map[:] = gradient
        self.stiffness_map = modulus / (self.thickness_map + 1e-5)
        
        mid_x = int(self.nx * 0.4)
        mid_range = int(self.nx * 0.15)
        self.stiffness_map[:, mid_x-mid_range : mid_x+mid_range] *= arch_stiff
        
        mask = np.ones_like(self.stiffness_map)
        if groove_type == "Horizontal Sipes": mask[:, ::3] = 0.2
        elif groove_type == "Grid Pattern": mask[:, ::4] = 0.2; mask[::4, :] = 0.2
        elif groove_type == "Honeycomb": mask[(self.X + self.Y) % 5 == 0] = 0.1
        self.stiffness_map *= mask

    def solve_static(self, weight_kg):
        target_force = weight_kg * 9.81
        penetration = 0
        foot_mask = self.foot_shape > 0.1
        for _ in range(500):
            local_compression = np.maximum((self.foot_shape * 10) - (10 - penetration), 0)
            reaction_pressure = self.stiffness_map * local_compression * foot_mask
            current_force = np.sum(reaction_pressure) * (self.dx**2)
            if current_force >= target_force:
                self.pressure_map = reaction_pressure
                break
            penetration += 0.1
        return self.pressure_map

    def solve_walking_step(self, weight_kg, gait_phase_pct):
        center_x = 5 + (gait_phase_pct * 20)
        focus_blob = np.exp(-((self.X - center_x)**2) / 15)
        base_pressure = self.solve_static(weight_kg)
        dynamic_pressure = base_pressure * focus_blob
        
        dynamic_factor = 1.2
        current_sum = np.sum(dynamic_pressure)
        applied_load_N = 0
        if current_sum > 0:
            target = weight_kg * 9.81 * dynamic_factor
            dynamic_pressure *= (target / (current_sum * (self.dx**2)))
            applied_load_N = target
        
        self.wear_map += dynamic_pressure * self.current_wear_factor
        return dynamic_pressure, applied_load_N