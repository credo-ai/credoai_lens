import ipywidgets as widgets
import pandas as pd
from IPython.display import display


class PolicyChecklist:
    def __init__(self, controls):
        self.controls = controls
        self.progress = self._create_progress()

    def _checkboxes_to_df(self, checkboxes):
        return pd.DataFrame(
            [
                {
                    "key": c.key,
                    "section": c.section,
                    "label": c.label,
                    "value": c.value,
                    "description": c.content,
                }
                for c in checkboxes
            ]
        )

    def _create_progress(self):
        return widgets.IntProgress(
            value=0,
            min=0,
            max=sum([len(v) for v in self.controls.values()]),
            description="<b>Checklist Progress:</b>",
            style={"bar_color": "green", "description_width": "150px"},
            orientation="horizontal",
            layout={"width": "50%"},
        )

    def _style_question(self, q):
        split = q.split(":", maxsplit=1)
        return f"<b>{split[0]}</b>: {split[1]}"

    def _segment(self, q):
        key = q.split()[0]
        label = q.split(":")[0][len(key) :]
        return key, label

    # for progress bar change
    def _get_progress_updater_fun(self):
        def fun(change):
            change = change["new"]
            if change:
                self.progress.value += 1
            else:
                self.progress.value -= 1

        return fun

    def create_checklist(self):
        """Create checklist from controls

        from: https://stackoverflow.com/questions/41469554/python-jupyter-notebook-create-dynamic-checklist-user-input-that-runs-code/59240442#59240442
        """
        widget_list = [widgets.HTML(f"<h1>Policy Checklist</h1>")]  # , self.progress]
        all_checkboxes = []
        for title, contents in self.controls.items():
            checkboxes = [widgets.Checkbox() for i in range(len(contents))]
            all_checkboxes += checkboxes
            for i, c in enumerate(checkboxes):
                c.description = self._style_question(contents[i])
                c.style = {
                    "description_width": "0px"
                }  # sets the width to the left of the check box
                c.layout.width = (
                    "max-content"  # sets the overall width check box widget
                )
                c.section = title
                c.key, c.label = self._segment(contents[i])
                c.content = contents[i]
                c.observe(self._get_progress_updater_fun(), names=["value"])
            widget_list += [
                widgets.HTML(f'<h3>{title.replace("_", " ").title()}</h3>')
            ] + checkboxes
        ui = widgets.VBox(widget_list)
        display(ui)
        return all_checkboxes
