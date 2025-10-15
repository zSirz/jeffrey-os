import os

import yaml
from core.ia_pricing import estimate_cost
from kivy.metrics import dp
from kivy.properties import DictProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "ia_registry.yaml")


class IAOverlay(BoxLayout):
    categorized_models = DictProperty({})
    total_estimated_cost = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_and_group_models()

    def load_and_group_models(self):
        if not os.path.exists(REGISTRY_PATH):
            return

        with open(REGISTRY_PATH) as file:
            try:
                registry = yaml.safe_load(file)
                models = registry.get("authorized_models", [])
                grouped = {}

                for model in models:
                    category = model.get("category", "autres")
                    if category not in grouped:
                        grouped[category] = []
                    grouped[category].append(model)

                self.categorized_models = grouped

                # Calcul du coût estimé total
                total = 0.0
                for models in grouped.values():
                    for model in models:
                        if model.get("enabled", False):
                            total += estimate_cost(model.get("name", ""), token_count=800)
                self.total_estimated_cost = total

            except yaml.YAMLError as e:
                print(f"Erreur YAML : {e}")

    def toggle_detailed_costs(self):
        box = self.ids.cost_details_box
        if box.opacity == 0:
            box.clear_widgets()
            for category, models in self.categorized_models.items():
                for model in models:
                    if model.get("enabled"):
                        cost = estimate_cost(model.get("name", ""), token_count=800)
                        label = Label(
                            text=f"{model.get('name')} : {cost:.3f} $",
                            size_hint_y=None,
                            height=dp(24),
                            color=(0.9, 0.9, 1, 1),
                        )
                        box.add_widget(label)
            box.opacity = 1
            box.height = len(box.children) * dp(26) + dp(10)
        else:
            box.opacity = 0
            box.height = 0

    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.switch import Switch

    def save_state_and_close(self):
        """
        Met à jour le fichier ia_registry.yaml avec les nouveaux états des switches.
        """
        new_registry = []

        for category, models in self.categorized_models.items():
            for model in models:
                name = model.get("name")
                # On recherche le switch correspondant dans l'interface
                for child in self.ids.content_box.children:
                    if isinstance(child, BoxLayout) and len(child.children) == 2:
                        label_widget, switch_widget = child.children[::-1]
                        if getattr(label_widget, "text", "") == name and isinstance(switch_widget, Switch):
                            new_registry.append(
                                {
                                    "name": name,
                                    "enabled": switch_widget.active,
                                    "category": model.get("category", "autres"),
                                }
                            )
                            break

        try:
            with open(REGISTRY_PATH, "w") as file:
                yaml.dump({"authorized_models": new_registry}, file)
        except Exception as e:
            print(f"[IAOverlay] Erreur d'enregistrement du registre IA : {e}")

        self.dismiss_overlay()

    def dismiss_overlay(self):
        self.parent.remove_widget(self)
