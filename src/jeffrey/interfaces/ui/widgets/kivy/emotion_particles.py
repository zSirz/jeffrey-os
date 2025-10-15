from collections import deque
from math import cos, radians, sin
from random import uniform

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, InstructionGroup, Line
from kivy.uix.widget import Widget


class EmotionParticle:
    def __init__(self, x, y, size, color, lifespan, shape="circle"):
        self.x = x
        self.y = y
        self.base_x = x  # Store initial position for oscillation
        self.base_y = y
        self.size = size
        self.base_size = size  # Base size for size oscillation
        self.color = color
        self.lifespan = lifespan
        self.age = 0
        self.angle = uniform(0, 360)  # Angle for circular/spiral movement
        self.rotation_speed = uniform(20, 60)  # Degrees per second for rotation
        self.oscillation_speed = uniform(2, 5)  # Speed of size oscillation
        self.oscillation_amplitude = size * 0.2  # Amplitude of size oscillation
        self.shape = shape

        self.graphic = InstructionGroup()
        self.color_instruction = Color(*self.color)
        self.graphic.add(self.color_instruction)

        # Create shape based on emotion type
        if self.shape == "heart":
            # Heart shape approximated by two circles and a triangle (simplified)
            # We'll draw a simple heart shape using Line instructions
            self.heart_line = Line(points=self._heart_points(self.x, self.y, self.size), close=True, width=1.5)
            self.graphic.add(self.heart_line)
        elif self.shape == "soft":
            # Soft shape: ellipse with smooth edges, slightly elongated
            self.ellipse = Ellipse(pos=(self.x, self.y), size=(self.size * 1.2, self.size))
            self.graphic.add(self.ellipse)
        else:
            # Default circle shape
            self.ellipse = Ellipse(pos=(self.x, self.y), size=(self.size, self.size))
            self.graphic.add(self.ellipse)

    def _heart_points(self, x, y, size):
        # Generate points for a simple heart shape centered at (x,y)
        # This is a rough approximation using bezier-like points
        # We'll create a polygon that looks like a heart
        s = size / 2
        points = [
            x,
            y + s * 0.5,
            x - s,
            y + s * 1.5,
            x - s * 1.5,
            y + s * 0.5,
            x - s * 1.5,
            y,
            x,
            y - s * 1.5,
            x + s * 1.5,
            y,
            x + s * 1.5,
            y + s * 0.5,
            x + s,
            y + s * 1.5,
        ]
        return points

    def update(self, dt):
        self.age += dt
        # Calculate normalized age (0 to 1)
        norm_age = self.age / self.lifespan

        # Pulsating alpha: oscillate alpha with a sine wave + fade out over lifespan
        pulse = 0.3 * sin(2 * 3.14159 * 2 * norm_age) + 0.7  # oscillates between 0.4 and 1.0 approx
        fade = max(0, 1 - norm_age)
        alpha = pulse * fade
        self.color_instruction.a = alpha

        # Spiral/oscillating movement: update angle and calculate new position
        self.angle += self.rotation_speed * dt
        rad = radians(self.angle)
        radius = 10 * (1 - norm_age)  # radius shrinks over time

        # Oscillate x position around base_x with a sine wave for horizontal movement
        self.x = self.base_x + radius * cos(rad)
        # Upward spiral movement with oscillation in y
        self.y = self.base_y + radius * sin(rad) + 10 * self.age

        # Size oscillation (grow/shrink slightly)
        size_oscillation = self.oscillation_amplitude * sin(self.oscillation_speed * self.age * 2 * 3.14159)
        current_size = max(1, self.base_size + size_oscillation)

        # Update shape position and size accordingly
        if self.shape == "heart":
            # Update heart points based on current position and size
            self.graphic.remove(self.heart_line)
            self.heart_line = Line(points=self._heart_points(self.x, self.y, current_size), close=True, width=1.5)
            self.graphic.add(self.heart_line)
        elif self.shape == "soft":
            self.ellipse.pos = (self.x - current_size * 0.6, self.y - current_size * 0.5)
            self.ellipse.size = (current_size * 1.2, current_size)
        else:
            self.ellipse.pos = (self.x - current_size / 2, self.y - current_size / 2)
            self.ellipse.size = (current_size, current_size)

        return self.age < self.lifespan


class EmotionParticleEmitter(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.particles = deque()
        self.canvas_particles = InstructionGroup()
        self.canvas.add(self.canvas_particles)
        Clock.schedule_interval(self.update_particles, 1 / 30.0)

    def emit_particle(self, emotion_type, intensity):
        # Base colors for emotions (HSV-like values for saturation adjustment)
        base_colors = {
            "happy": (1, 0.8, 0.2),
            "sad": (0.2, 0.4, 1),
            "angry": (1, 0.2, 0.2),
            "peaceful": (0.6, 1, 0.6),
            "in_love": (1, 0.4, 0.8),
        }

        # Function to adjust color saturation based on intensity
        def adjust_saturation(rgb, intensity):
            # Clamp intensity between 0 and 1
            intensity = max(0, min(1, intensity))
            r, g, b = rgb
            # Increase saturation by moving color away from gray (1,1,1)
            r = r + (r - 1) * (intensity - 0.5)
            g = g + (g - 1) * (intensity - 0.5)
            b = b + (b - 1) * (intensity - 0.5)
            # Clamp to [0,1]
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            return (r, g, b)

        base_color = base_colors.get(emotion_type, (1, 1, 1))
        color = adjust_saturation(base_color, intensity) + (1,)

        # Determine shape based on emotion type
        shape_map = {
            "in_love": "heart",
            "peaceful": "soft",
        }
        shape = shape_map.get(emotion_type, "circle")

        for _ in range(int(5 + intensity * 10)):
            x = self.center_x + uniform(-50, 50)
            y = self.center_y + uniform(-20, 20)
            size = uniform(5, 12) * intensity
            lifespan = uniform(0.5, 1.5)

            particle = EmotionParticle(x, y, size, color, lifespan, shape=shape)
            self.particles.append(particle)
            self.canvas_particles.add(particle.graphic)

    def update_particles(self, dt):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            if p.update(dt):
                self.particles.append(p)
            else:
                self.canvas_particles.remove(p.graphic)
