import csv
import json
import os
import platform
import subprocess
from datetime import datetime

from kivy.animation import Animation
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.list import IconLeftWidget, ThreeLineListItem
from kivymd.uix.snackbar import Snackbar
from reportlab.lib import colors

# PDF export dependencies
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ia_task_history.json")


class IAHistoryWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_history()

    def load_history(self, filter_model=None, sort_by=None):
        self.ids.history_list.clear_widgets()
        total_cost = 0.0
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []

        # Tri interactif selon sort_by
        if sort_by == "date":
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        elif sort_by == "cost":

            def safe_cost(entry):
                try:
                    return float(entry.get("estimated_cost", 0))
                except (ValueError, TypeError):
                    return 0

            history.sort(key=safe_cost, reverse=True)
        elif sort_by == "ia":
            history.sort(key=lambda x: ", ".join(x.get("models", [])).lower())

        # Correction de l'ordre d'affichage : toujours afficher les plus récentes en haut
        for entry in history[::-1]:
            models_list = entry.get("models", [])
            if filter_model and filter_model not in models_list:
                continue

            task = entry.get("task", "Tâche inconnue")
            models = ", ".join(models_list)
            date = entry.get("timestamp", "inconnue")
            cost = entry.get("estimated_cost", "?")

            try:
                total_cost += float(cost)
            except (ValueError, TypeError):
                pass

            try:
                formatted_date = datetime.fromisoformat(date).strftime("%d/%m %H:%M") if date != "inconnue" else date
            except Exception:
                formatted_date = date

            item = ThreeLineListItem(
                text=task,
                secondary_text=f"🧠 IA : {models}",
                tertiary_text=f"📅 {formatted_date}   💸 {cost} $",
            )
            item.add_widget(IconLeftWidget(icon="robot-outline"))
            self.ids.history_list.add_widget(item)

        self.ids.cost_label.text = f"💰 Coût total estimé : {total_cost:.2f} $"

    def filter_by_model(self, model_name):
        self.load_history(filter_model=model_name, sort_by="date")

    def filter_by_period(self, period):
        from datetime import datetime, timedelta

        self.ids.history_list.clear_widgets()
        total_cost = 0.0
        now = datetime.now()

        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []

        for entry in reversed(history):
            date_str = entry.get("timestamp")
            if not date_str:
                continue
            try:
                entry_date = datetime.fromisoformat(date_str)
            except ValueError:
                continue

            show = False
            if period == "Aujourd’hui":
                show = entry_date.date() == now.date()
            elif period == "Cette semaine":
                start_week = now - timedelta(days=now.weekday())
                show = entry_date.date() >= start_week.date()
            elif period == "Ce mois-ci":
                show = entry_date.year == now.year and entry_date.month == now.month
            else:
                show = True  # "Toutes" ou autre

            if show:
                task = entry.get("task", "Tâche inconnue")
                models = ", ".join(entry.get("models", []))
                cost = entry.get("estimated_cost", "?")

                try:
                    total_cost += float(cost)
                except (ValueError, TypeError):
                    pass

                formatted_date = entry_date.strftime("%d/%m %H:%M")

                item = ThreeLineListItem(
                    text=task,
                    secondary_text=f"🧠 IA : {models}",
                    tertiary_text=f"📅 {formatted_date}   💸 {cost} $",
                )
                item.add_widget(IconLeftWidget(icon="robot-outline"))
                self.ids.history_list.add_widget(item)

        self.ids.cost_label.text = f"💰 Coût total estimé : {total_cost:.2f} $"

    def export_to_csv(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ids.spinner_csv.active = True
        self.ids.spinner_csv.opacity = 1
        self.ids.check_csv.opacity = 0
        export_path = os.path.join(
            os.path.dirname(__file__), "..", "exports", f"ia_task_history_export_{timestamp}.csv"
        )
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        if not os.path.exists(HISTORY_PATH):
            return

        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

        with open(export_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Date", "Tâche", "IA utilisée(s)", "Coût estimé"])

            for entry in history:
                date = entry.get("timestamp", "")
                task = entry.get("task", "")
                models = ", ".join(entry.get("models", []))
                cost = entry.get("estimated_cost", "")
                writer.writerow([date, task, models, cost])

        print(f"[✔] Export CSV terminé : {export_path}")
        self.open_export_folder()
        Snackbar(text="✅ Export CSV terminé avec succès").open()
        if hasattr(App.get_running_app(), "jeffrey"):
            App.get_running_app().jeffrey.say_with_emotion(
                base_phrases=["L'export CSV est prêt ! Tu peux le consulter."], context="success"
            )
        # Animation de rebond du bouton export
        try:
            export_button = self.ids.get("export_csv_button", None)
            if export_button:
                export_button.disabled = True
                Animation(scale=1.05, d=0.1).start(export_button)
                Animation(scale=1.0, d=0.1).start(export_button)
                export_button.disabled = False
            # Hide spinner and show checkmark icon
            self.ids.spinner_csv.active = False
            self.ids.spinner_csv.opacity = 0
            self.ids.check_csv.opacity = 1
            check_icon = self.ids.check_csv
            check_icon.angle = 0
            Animation(opacity=1, angle=360, d=0.5, t="out_quad").start(check_icon)
        except Exception as e:
            print(f"[Animation] Erreur : {e}")

    def export_to_pdf(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ids.spinner_pdf.active = True
        self.ids.spinner_pdf.opacity = 1
        self.ids.check_pdf.opacity = 0
        export_path = os.path.join(
            os.path.dirname(__file__), "..", "exports", f"ia_task_history_export_{timestamp}.pdf"
        )
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        if not os.path.exists(HISTORY_PATH):
            return

        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

        doc = SimpleDocTemplate(export_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        title = Paragraph("🧠 Historique des Tâches IA", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 12))

        data = [["Date", "Tâche", "IA utilisée(s)", "Coût estimé"]]
        total_cost = 0.0

        for entry in history:
            date = entry.get("timestamp", "")
            task = entry.get("task", "")
            models = ", ".join(entry.get("models", []))
            cost = entry.get("estimated_cost", "?")
            try:
                total_cost += float(cost)
            except:
                pass
            data.append([date, task, models, cost])

        table = Table(data, colWidths=[90, 160, 120, 60])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 12))

        summary = Paragraph(f"<b>💰 Coût total estimé :</b> {total_cost:.2f} $", styles["Normal"])
        elements.append(summary)

        doc.build(elements)
        print(f"[✔] Export PDF terminé : {export_path}")
        self.open_export_folder()
        Snackbar(text="✅ Export PDF terminé avec succès").open()
        if hasattr(App.get_running_app(), "jeffrey"):
            App.get_running_app().jeffrey.say_with_emotion(
                base_phrases=["J'ai terminé le PDF ! Il t'attend dans les exports."],
                context="success",
            )
        # Animation de rebond du bouton export
        try:
            export_button = self.ids.get("export_pdf_button", None)
            if export_button:
                export_button.disabled = True
                Animation(scale=1.05, d=0.1).start(export_button)
                Animation(scale=1.0, d=0.1).start(export_button)
                export_button.disabled = False
            # Hide spinner and show checkmark icon
            self.ids.spinner_pdf.active = False
            self.ids.spinner_pdf.opacity = 0
            self.ids.check_pdf.opacity = 1
            check_icon = self.ids.check_pdf
            check_icon.angle = 0
            Animation(opacity=1, angle=360, d=0.5, t="out_quad").start(check_icon)
        except Exception as e:
            print(f"[Animation] Erreur : {e}")

    def open_export_folder(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
