import tkinter as tk
from tkinter import ttk, font as tkfont
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# ── train model on startup
def load_and_train():
    ds = pd.read_excel("energy_efficiency.xlsx")
    ds.columns = [
        "relative_compactness", "surface_area", "wall_area", "roof_area",
        "overall_height", "orientation", "glazing_area",
        "glazing_area_distribution", "heating_load", "cooling_load"
    ]
    ds_clean = ds.drop(["orientation", "surface_area", "roof_area",
                         "overall_height", "glazing_area_distribution",
                         "cooling_load"], axis=1)
    ds_eng = ds_clean.copy()
    ds_eng["compactness_per_height"] = ds["relative_compactness"] / ds["overall_height"]

    X = ds_eng.drop("heating_load", axis=1)
    y = ds_eng["heating_load"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    best_r2, best_depth = 0, 0
    for d in range(1, 10):
        m = DecisionTreeRegressor(max_depth=d, random_state=42)
        cv = cross_val_score(m, X_train, y_train, cv=5, scoring="r2")
        if cv.mean() > best_r2:
            best_r2 = cv.mean()
            best_depth = d

    model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)
    return model, best_depth

# ── colour scheme
C = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "card":     "#1e2530",
    "border":   "#30363d",
    "accent":   "#58a6ff",
    "green":    "#3fb950",
    "yellow":   "#d29922",
    "orange":   "#f0883e",
    "red":      "#f85149",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "input_bg": "#0d1117",
    "hover":    "#21262d",
    "tag_bg":   "#1f2a3a",
    "tag_fg":   "#58a6ff",
}

# ── field definitions
FIELDS = {
    "relative_compactness": {
        "label":   "Relative Compactness",
        "type":    "dropdown",
        "options": ["0.62", "0.64", "0.66", "0.69", "0.71", "0.74",
                    "0.76", "0.79", "0.82", "0.86", "0.90", "0.98"],
        "unit":    "ratio",
        "desc":    "Shape efficiency of the building (higher = more compact)",
        "tips": {
            "0.62": "Very spread out building — lots of exposed surface, higher heat loss",
            "0.64": "Spread-out building — moderate surface exposure",
            "0.66": "Slightly spread — still significant external surface area",
            "0.69": "Below average compactness — moderate energy performance",
            "0.71": "Average compactness — balanced shape",
            "0.74": "Slightly above average — good shape efficiency",
            "0.76": "Good compactness — reduced surface-to-volume ratio",
            "0.79": "High compactness — efficient heat retention",
            "0.82": "Very high compactness — minimal heat loss through shape",
            "0.86": "Excellent compactness — near-optimal building form",
            "0.90": "Near-perfect cube-like shape — very low surface exposure",
            "0.98": "Maximum compactness — cube-shaped, lowest possible heat loss",
        }
    },
    "wall_area": {
        "label":   "Wall Area",
        "type":    "dropdown",
        "options": ["245.0", "269.5", "294.0", "318.5", "343.0",
                    "367.5", "392.0", "416.5"],
        "unit":    "m²",
        "desc":    "Total external wall surface area of the building",
        "tips": {
            "245.0": "Smallest wall area — compact building with minimal exposed walls",
            "269.5": "Small wall area — low heat loss through walls",
            "294.0": "Below average wall area — efficient wall footprint",
            "318.5": "Average wall area — typical residential building",
            "343.0": "Slightly large wall area — more insulation may be needed",
            "367.5": "Large wall area — significant heat loss pathway",
            "392.0": "Very large wall area — high insulation requirement",
            "416.5": "Maximum wall area — greatest wall-driven heat loss in dataset",
        }
    },
    "glazing_area": {
        "label":   "Glazing Area",
        "type":    "dropdown",
        "options": ["0.0", "0.10", "0.25", "0.40"],
        "unit":    "ratio",
        "desc":    "Window area as fraction of total floor area",
        "tips": {
            "0.0":  "No windows — zero glazing, best thermal performance",
            "0.10": "10% glazing — minimal windows, low heat loss through glass",
            "0.25": "25% glazing — moderate windows, balanced light and efficiency",
            "0.40": "40% glazing — large windows, significant heat loss pathway",
        }
    },
    "overall_height": {
        "label":   "Overall Height",
        "type":    "dropdown",
        "options": ["3.5", "7.0"],
        "unit":    "m",
        "desc":    "Total height of the building (affects compactness ratio)",
        "tips": {
            "3.5": "Single storey — lower height, compactness_per_height is higher, less heating needed",
            "7.0": "Double storey — taller building, lower compactness ratio, more heating needed",
        }
    },
    "glazing_area_distribution": {
        "label":   "Glazing Distribution",
        "type":    "dropdown",
        "options": ["0", "1", "2", "3", "4", "5"],
        "unit":    "",
        "desc":    "Where windows are placed around the building",
        "tips": {
            "0": "No glazing — no windows at all",
            "1": "North-facing only — least sunlight, minimal solar gain",
            "2": "East-facing only — morning light, moderate solar gain",
            "3": "South-facing only — maximum sunlight, higher solar gain",
            "4": "West-facing only — afternoon light, moderate-high solar gain",
            "5": "Uniform distribution — windows on all sides evenly",
        }
    },
}

class HeatingLoadGUI:
    def __init__(self, root, model, best_depth):
        self.root    = root
        self.model   = model
        self.depth   = best_depth
        self.vars    = {}
        self.tip_labels = {}
        self._build_ui()

    def _build_ui(self):
        r = self.root
        r.title("Heating Load Predictor")
        r.configure(bg=C["bg"])
        r.resizable(True, True)
        r.minsize(680, 720)

        # ── scrollable canvas setup
        canvas = tk.Canvas(r, bg=C["bg"], highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(r, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.frame = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=self.frame, anchor="nw")

        def on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        self.frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        wrap = self.frame
        wrap.configure(padx=28, pady=20)

        # ── header
        hdr = tk.Frame(wrap, bg=C["bg"])
        hdr.pack(fill="x", pady=(0, 22))

        tk.Label(hdr, text="Heating Load Predictor",
                 font=("Helvetica", 20, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(anchor="w")
        tk.Label(hdr,
                 text=f"Decision Tree  ·  depth={self.depth}  ·  Energy Efficiency Dataset",
                 font=("Helvetica", 10),
                 bg=C["bg"], fg=C["muted"]).pack(anchor="w", pady=(2, 0))

        sep = tk.Frame(wrap, bg=C["border"], height=1)
        sep.pack(fill="x", pady=(0, 20))

        # ── input fields
        for key, cfg in FIELDS.items():
            self._build_field(wrap, key, cfg)

        # ── spacer
        tk.Frame(wrap, bg=C["bg"], height=10).pack()

        # ── buttons
        btn_row = tk.Frame(wrap, bg=C["bg"])
        btn_row.pack(fill="x", pady=(4, 20))

        predict_btn = tk.Button(
            btn_row, text="  Predict Heating Load  ",
            font=("Helvetica", 12, "bold"),
            bg=C["accent"], fg="#0d1117",
            relief="flat", bd=0, cursor="hand2",
            activebackground="#79b8ff", activeforeground="#0d1117",
            command=self._predict, padx=10, pady=8
        )
        predict_btn.pack(side="left", ipadx=4)

        clear_btn = tk.Button(
            btn_row, text="  Clear All  ",
            font=("Helvetica", 11),
            bg=C["card"], fg=C["muted"],
            relief="flat", bd=0, cursor="hand2",
            activebackground=C["hover"], activeforeground=C["text"],
            command=self._clear, padx=10, pady=8
        )
        clear_btn.pack(side="left", padx=(12, 0), ipadx=4)

        # ── result card
        sep2 = tk.Frame(wrap, bg=C["border"], height=1)
        sep2.pack(fill="x", pady=(0, 16))

        self.result_card = tk.Frame(wrap, bg=C["panel"],
                                    relief="flat", bd=0,
                                    highlightthickness=1,
                                    highlightbackground=C["border"])
        self.result_card.pack(fill="x", pady=(0, 10))

        inner = tk.Frame(self.result_card, bg=C["panel"], padx=20, pady=18)
        inner.pack(fill="x")

        tk.Label(inner, text="Predicted Heating Load",
                 font=("Helvetica", 11),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w")

        self.result_value = tk.Label(inner, text="—",
                                     font=("Helvetica", 38, "bold"),
                                     bg=C["panel"], fg=C["accent"])
        self.result_value.pack(anchor="w")

        self.result_unit = tk.Label(inner, text="kWh / m²",
                                    font=("Helvetica", 11),
                                    bg=C["panel"], fg=C["muted"])
        self.result_unit.pack(anchor="w")

        self.demand_label = tk.Label(inner, text="",
                                     font=("Helvetica", 12, "bold"),
                                     bg=C["panel"], fg=C["text"])
        self.demand_label.pack(anchor="w", pady=(8, 0))

        # ── interpretation card
        self.interp_frame = tk.Frame(wrap, bg=C["panel"],
                                     relief="flat", bd=0,
                                     highlightthickness=1,
                                     highlightbackground=C["border"])
        self.interp_frame.pack(fill="x", pady=(0, 20))

        interp_inner = tk.Frame(self.interp_frame, bg=C["panel"], padx=20, pady=14)
        interp_inner.pack(fill="x")

        tk.Label(interp_inner, text="Interpretation",
                 font=("Helvetica", 10, "bold"),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", pady=(0, 6))

        self.interp_text = tk.Label(interp_inner,
                                    text="Fill in all fields above and click Predict.",
                                    font=("Helvetica", 11),
                                    bg=C["panel"], fg=C["muted"],
                                    wraplength=580, justify="left")
        self.interp_text.pack(anchor="w")

    def _build_field(self, parent, key, cfg):
        card = tk.Frame(parent, bg=C["card"],
                        highlightthickness=1,
                        highlightbackground=C["border"])
        card.pack(fill="x", pady=(0, 10))

        inner = tk.Frame(card, bg=C["card"], padx=16, pady=12)
        inner.pack(fill="x")

        # label row
        label_row = tk.Frame(inner, bg=C["card"])
        label_row.pack(fill="x")

        tk.Label(label_row, text=cfg["label"],
                 font=("Helvetica", 12, "bold"),
                 bg=C["card"], fg=C["text"]).pack(side="left")

        if cfg["unit"]:
            tk.Label(label_row, text=cfg["unit"],
                     font=("Helvetica", 10),
                     bg=C["card"], fg=C["tag_fg"],
                     padx=6, pady=1,
                     relief="flat").pack(side="left", padx=(8, 0))

        # description
        tk.Label(inner, text=cfg["desc"],
                 font=("Helvetica", 10),
                 bg=C["card"], fg=C["muted"],
                 anchor="w").pack(fill="x", pady=(2, 8))

        # dropdown
        var = tk.StringVar(value="— select —")
        self.vars[key] = var

        combo_frame = tk.Frame(inner, bg=C["input_bg"],
                               highlightthickness=1,
                               highlightbackground=C["border"])
        combo_frame.pack(fill="x")

        combo = ttk.Combobox(combo_frame, textvariable=var,
                             values=["— select —"] + cfg["options"],
                             state="readonly", font=("Helvetica", 12),
                             height=12)
        combo.pack(fill="x", ipady=4)

        # tip label
        tip = tk.Label(inner, text="",
                       font=("Helvetica", 10),
                       bg=C["card"], fg=C["accent"],
                       wraplength=560, justify="left", anchor="w")
        tip.pack(fill="x", pady=(6, 0))
        self.tip_labels[key] = tip

        # bind selection
        def on_select(event, k=key, t=tip, c=cfg):
            val = self.vars[k].get()
            if val in c["tips"]:
                t.config(text=f"  {c['tips'][val]}", fg=C["accent"])
            else:
                t.config(text="")

        combo.bind("<<ComboboxSelected>>", on_select)

        # style combobox
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=C["input_bg"],
                        background=C["card"],
                        foreground=C["text"],
                        selectbackground=C["hover"],
                        selectforeground=C["text"],
                        arrowcolor=C["muted"],
                        bordercolor=C["border"],
                        lightcolor=C["border"],
                        darkcolor=C["border"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", C["input_bg"])],
                  foreground=[("readonly", C["text"])],
                  background=[("readonly", C["card"])])

    def _predict(self):
        vals = {}
        for key in FIELDS:
            v = self.vars[key].get()
            if v == "— select —" or v == "":
                self._show_error(f"Please select a value for: {FIELDS[key]['label']}")
                return
            try:
                vals[key] = float(v)
            except ValueError:
                self._show_error(f"Invalid value for {FIELDS[key]['label']}")
                return

        rc  = vals["relative_compactness"]
        wa  = vals["wall_area"]
        ga  = vals["glazing_area"]
        oh  = vals["overall_height"]
        cph = rc / oh

        user_df = pd.DataFrame([[rc, wa, ga, cph]],
                               columns=["relative_compactness", "wall_area",
                                        "glazing_area", "compactness_per_height"])

        pred = self.model.predict(user_df)[0]

        self.result_value.config(text=f"{pred:.2f}")

        if pred < 12:
            colour  = C["green"]
            demand  = "Low Demand"
            interp  = (
                f"A predicted heating load of {pred:.2f} kWh/m² is low — well below the dataset "
                f"average of 22.3 kWh/m². This building has high compactness ({rc}) and limited "
                f"glazing ({ga*100:.0f}% of floor area), meaning little energy is needed to keep "
                f"it warm. Excellent energy efficiency."
            )
        elif pred < 22:
            colour  = C["accent"]
            demand  = "Moderate Demand"
            interp  = (
                f"A predicted load of {pred:.2f} kWh/m² is below the dataset average (22.3 kWh/m²). "
                f"The building shape (compactness {rc}) and wall area ({wa} m²) are reasonably "
                f"efficient. Modest insulation improvements could push this into the low-demand range."
            )
        elif pred < 32:
            colour  = C["orange"]
            demand  = "High Demand"
            interp  = (
                f"At {pred:.2f} kWh/m², this building is above the dataset average (22.3 kWh/m²). "
                f"The wall area of {wa} m² and glazing of {ga*100:.0f}% are significant heat loss "
                f"pathways. Better insulation, double glazing, or a more compact design would help."
            )
        else:
            colour  = C["red"]
            demand  = "Very High Demand"
            interp  = (
                f"At {pred:.2f} kWh/m² this is among the highest energy demands in the dataset. "
                f"The combination of large wall area ({wa} m²), high glazing ({ga*100:.0f}%), and "
                f"low compactness ({rc}) creates significant heat loss. Major retrofitting or "
                f"redesign of the building shape is recommended."
            )

        self.demand_label.config(text=f"  {demand}", fg=colour)
        self.result_value.config(fg=colour)
        self.interp_text.config(text=interp, fg=C["text"])

    def _clear(self):
        for key in self.vars:
            self.vars[key].set("— select —")
        for tip in self.tip_labels.values():
            tip.config(text="")
        self.result_value.config(text="—", fg=C["accent"])
        self.demand_label.config(text="")
        self.interp_text.config(
            text="Fill in all fields above and click Predict.",
            fg=C["muted"])

    def _show_error(self, msg):
        win = tk.Toplevel(self.root)
        win.title("Missing Input")
        win.configure(bg=C["panel"])
        win.resizable(False, False)
        win.geometry("380x130")
        tk.Label(win, text=msg,
                 font=("Helvetica", 11),
                 bg=C["panel"], fg=C["text"],
                 wraplength=320, justify="center").pack(pady=28)
        tk.Button(win, text="OK", command=win.destroy,
                  bg=C["accent"], fg="#0d1117",
                  font=("Helvetica", 11, "bold"),
                  relief="flat", padx=22, pady=6,
                  cursor="hand2").pack()


# ── run
if __name__ == "__main__":
    print("Training model, please wait...")
    model, depth = load_and_train()
    print(f"Model ready (depth={depth})")

    root = tk.Tk()
    app  = HeatingLoadGUI(root, model, depth)
    root.mainloop()