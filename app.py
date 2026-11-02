"""Multi-Annotator Sentiment: a Tkinter annotator window and a Stats & Classifier window.

    python app.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import analysis
import db

PLACEHOLDER = "Select label"
DISPLAY = {label: label.capitalize() for label in db.LABELS}
FROM_DISPLAY = {v: k for k, v in DISPLAY.items()}
FILTERS = {"All records": None, "Imported": "import", "New (this app)": "ui"}


def scale(widget: tk.Misc) -> float:
    """Screen scaling relative to 96 dpi, so pixel sizes stay right on high-DPI displays."""
    return max(1.0, widget.winfo_fpixels("1i") / 96)


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


class AnnotatorApp:
    def __init__(self, root: tk.Tk, conn: sqlite3.Connection) -> None:
        self.root = root
        self.conn = conn
        self.current: str | None = None
        self.saved = False
        self.session_saved = 0
        self.stats: StatsWindow | None = None

        root.title("Multi-Annotator Sentiment")
        root.minsize(int(520 * scale(root)), 0)
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Current sentence:", style="Heading.TLabel").pack(anchor="w")
        self.sentence_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.sentence_var, style="Sentence.TLabel", wraplength=int(480 * scale(root)),
                  justify="left").pack(anchor="w", pady=(4, 14))

        grid = ttk.Frame(frame)
        grid.pack(anchor="w")
        self.vars: list[tk.StringVar] = []
        self.boxes: list[ttk.Combobox] = []
        for i, name in enumerate(db.ANNOTATORS):
            ttk.Label(grid, text=f"Annotator {i + 1}:").grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=PLACEHOLDER)
            box = ttk.Combobox(grid, textvariable=var, values=list(DISPLAY.values()), state="readonly", width=16)
            box.grid(row=i, column=1, sticky="w", padx=(10, 0), pady=3)
            box.bind("<<ComboboxSelected>>", lambda _e: self.update_state())
            var.trace_add("write", lambda *_a: self.update_state())
            self.vars.append(var)
            self.boxes.append(box)

        buttons = ttk.Frame(frame)
        buttons.pack(anchor="w", pady=(16, 6))
        self.save_button = ttk.Button(buttons, text="Save annotation", command=self.save, style="Accent.TButton")
        self.save_button.pack(side="left")
        self.generate_button = ttk.Button(buttons, text="Generate sentence", command=self.generate)
        self.generate_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Show stats", command=self.open_stats).pack(side="left")

        self.status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.status_var, style="Muted.TLabel", wraplength=int(480 * scale(root)),
                  justify="left").pack(anchor="w", pady=(6, 0))
        self.count_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.count_var, style="Muted.TLabel").pack(anchor="w")

        self.generate()

    # Annotation flow

    def labels(self) -> list[str | None]:
        return [FROM_DISPLAY.get(var.get()) for var in self.vars]

    def missing(self) -> list[int]:
        return [i + 1 for i, label in enumerate(self.labels()) if label is None]

    def update_state(self) -> None:
        if self.saved or self.current is None:
            self.save_button.state(["disabled"])
            return
        missing = self.missing()
        self.save_button.state(["!disabled"] if not missing else ["disabled"])
        if not missing:
            self.status_var.set("All three annotators have chosen. Save to store this record.")
        else:
            names = ", ".join(f"Annotator {i}" for i in missing)
            self.status_var.set(f"Waiting for: {names}.")

    def generate(self) -> None:
        self.current = db.random_sentence(self.conn, exclude=self.current)
        self.saved = False
        self.sentence_var.set(self.current or "The database has no sentences. Run import_data.py first.")
        for var, box in zip(self.vars, self.boxes):
            box.state(["!disabled", "readonly"])
            var.set(PLACEHOLDER)
        self.update_counts()
        self.update_state()

    def save(self) -> int | None:
        labels = self.labels()
        if self.saved or self.current is None or None in labels:
            return None
        try:
            rec_id = db.add_annotation(self.conn, self.current, labels)
        except (sqlite3.Error, ValueError) as err:
            messagebox.showerror("Could not save", str(err), parent=self.root)
            return None
        self.saved = True
        self.session_saved += 1
        for box in self.boxes:
            box.state(["disabled"])
        self.status_var.set(f"Saved as record #{rec_id}. Saved records can't be changed. "
                            "Generate a sentence to continue.")
        self.update_counts()
        self.update_state()
        if self.stats is not None and self.stats.alive():
            self.stats.refresh()
        return rec_id

    def update_counts(self) -> None:
        self.count_var.set(f"Saved this session: {self.session_saved} · Records in database: {db.count(self.conn)}")

    def open_stats(self) -> None:
        if self.stats is not None and self.stats.alive():
            self.stats.window.deiconify()
            self.stats.window.lift()
            self.stats.refresh()
            return
        self.stats = StatsWindow(self.root, self.conn)


class StatsWindow:
    RECORD_COLUMNS = (
        ("id", "Id", 60, "e"), ("text", "Text", 270, "w"), ("a1", "Annotator 1", 90, "w"),
        ("a2", "Annotator 2", 90, "w"), ("a3", "Annotator 3", 90, "w"), ("resolved", "Resolved", 85, "w"),
        ("prediction", "Prediction", 85, "w"), ("confidence", "Confidence", 85, "e"),
        ("agreement", "Agreement", 85, "e"), ("source", "Source", 70, "w"),
    )
    SENTENCE_COLUMNS = (
        ("text", "Text", 290, "w"), ("records", "Records", 70, "e"),
        *((label, DISPLAY[label], 80, "e") for label in db.LABELS),
        ("majority", "Majority", 90, "w"), ("agreement", "Avg agreement", 110, "e"),
    )

    def __init__(self, parent: tk.Tk, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.result: analysis.ClassifierResult | None = None
        self.records: list[db.Record] = []
        self.window = tk.Toplevel(parent)
        self.window.title("Stats & Classifier")
        factor = scale(self.window)
        self.window.geometry(f"{int(1180 * factor)}x{int(760 * factor)}")
        self.window.minsize(int(900 * factor), int(560 * factor))

        header = ttk.Frame(self.window, padding=(12, 10, 12, 4))
        header.pack(fill="x")
        self.summary_vars: dict[str, tk.StringVar] = {}
        for i, (key, title) in enumerate((
            ("total", "Total records"), ("unique", "Unique texts"), ("agreement", "Average agreement"),
            ("ties", "Ties"), ("accuracy", "Classifier accuracy"), ("unseen", "Unseen-sentence accuracy"),
        )):
            cell = ttk.Frame(header)
            cell.grid(row=0, column=i, sticky="w", padx=(0, 26))
            ttk.Label(cell, text=title, style="Muted.TLabel").pack(anchor="w")
            var = tk.StringVar(value="…")
            ttk.Label(cell, textvariable=var, style="Metric.TLabel").pack(anchor="w")
            self.summary_vars[key] = var

        controls = ttk.Frame(self.window, padding=(12, 4))
        controls.pack(fill="x")
        ttk.Button(controls, text="Rerun classifier", command=self.refresh, style="Accent.TButton").pack(side="left")
        ttk.Label(controls, text="Seed").pack(side="left", padx=(14, 4))
        self.seed_var = tk.StringVar(value="42")
        ttk.Spinbox(controls, from_=0, to=9999, textvariable=self.seed_var, width=6).pack(side="left")
        ttk.Label(controls, text="Show").pack(side="left", padx=(14, 4))
        self.filter_var = tk.StringVar(value="All records")
        show = ttk.Combobox(controls, textvariable=self.filter_var, values=list(FILTERS), state="readonly", width=15)
        show.pack(side="left")
        show.bind("<<ComboboxSelected>>", lambda _e: self.fill_records())
        ttk.Button(controls, text="Export CSV", command=self.export).pack(side="right")
        self.status_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.status_var, style="Muted.TLabel").pack(side="right", padx=10)

        body = ttk.PanedWindow(self.window, orient="vertical")
        body.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        tabs = ttk.Notebook(body)
        self.record_tree = self._tree(tabs, self.RECORD_COLUMNS)
        self.sentence_tree = self._tree(tabs, self.SENTENCE_COLUMNS)
        tabs.add(self.record_tree.master, text="Records")
        tabs.add(self.sentence_tree.master, text="By sentence")
        body.add(tabs, weight=3)

        report_frame = ttk.LabelFrame(body, text="Classification report", padding=6)
        self.report = tk.Text(report_frame, height=16, wrap="none", font=("Consolas", 10), relief="flat",
                              borderwidth=0, background="#fbfbfb")
        report_scroll = ttk.Scrollbar(report_frame, orient="vertical", command=self.report.yview)
        self.report.configure(yscrollcommand=report_scroll.set)
        self.report.pack(side="left", fill="both", expand=True)
        report_scroll.pack(side="right", fill="y")
        body.add(report_frame, weight=2)

        self.refresh()

    def _tree(self, parent: ttk.Notebook, columns) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=12)
        factor = scale(parent)
        for key, title, width, anchor in columns:
            tree.heading(key, text=title, anchor=anchor)
            tree.column(key, width=int(width * factor), minwidth=int(40 * factor), anchor=anchor,
                        stretch=key == "text")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        tree.tag_configure("new", background="#fff4c2")
        tree.tag_configure("tie", foreground="#8a6d00")
        return tree

    def alive(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def seed(self) -> int:
        try:
            return max(0, int(self.seed_var.get()))
        except ValueError:
            self.seed_var.set("42")
            return 42

    def refresh(self) -> None:
        """Reload every record, retrain the classifier and redraw."""
        self.window.config(cursor="watch")
        self.window.update_idletasks()
        try:
            self.records = db.records(self.conn)
            summary = analysis.summarize(self.records)
            self.result = analysis.train_classifier(self.records, seed=self.seed())
        finally:
            self.window.config(cursor="")

        v = self.summary_vars
        v["total"].set(str(summary.total))
        v["unique"].set(str(summary.unique_texts))
        v["agreement"].set(pct(summary.average_agreement))
        v["ties"].set(str(summary.ties))
        v["accuracy"].set(pct(self.result.accuracy))
        v["unseen"].set(pct(self.result.unseen_accuracy))

        self.fill_records()
        self.fill_sentences()
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", analysis.describe(self.result))
        self.report.configure(state="disabled")
        self.status_var.set(f"Trained on {self.result.train_size} records with seed {self.result.seed}"
                            if self.result.ok else "Classifier not trained")

    def fill_records(self) -> None:
        tree = self.record_tree
        tree.delete(*tree.get_children())
        source = FILTERS.get(self.filter_var.get())
        predictions = self.result.predictions if self.result else {}
        for r in self.records:
            if source and r.source != source:
                continue
            resolved, agreement = analysis.resolve(r.labels)
            prediction, confidence = predictions.get(r.id, ("—", None))
            tags = ("new",) if r.source == "ui" else ()
            if resolved == analysis.TIE:
                tags += ("tie",)
            tree.insert("", "end", iid=str(r.id), tags=tags, values=(
                r.id, r.text, *r.labels, resolved, prediction, pct(confidence), pct(agreement),
                "new" if r.source == "ui" else "import",
            ))

    def fill_sentences(self) -> None:
        tree = self.sentence_tree
        tree.delete(*tree.get_children())
        for row in analysis.by_sentence(self.records):
            tree.insert("", "end", tags=("tie",) if row.majority == analysis.TIE else (), values=(
                row.text, row.records, *(row.votes[label] for label in db.LABELS), row.majority,
                pct(row.average_agreement),
            ))

    def export(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
                                            initialfile="annotation_stats.csv", filetypes=[("CSV", "*.csv")])
        if path:
            count = export_csv(Path(path), self.records, self.result)
            self.status_var.set(f"Exported {count} rows to {Path(path).name}")


def export_csv(path: Path, records: list[db.Record], result: analysis.ClassifierResult | None) -> int:
    predictions = result.predictions if result else {}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", *db.ANNOTATORS, "resolved", "agreement", "prediction", "confidence", "source"])
        for r in records:
            resolved, agreement = analysis.resolve(r.labels)
            prediction, confidence = predictions.get(r.id, ("", None))
            writer.writerow([r.id, r.text, *r.labels, resolved, f"{agreement:.4f}", prediction,
                             "" if confidence is None else f"{confidence:.4f}", r.source])
    return len(records)


def apply_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    base = ("Segoe UI", 10)
    style.configure("TLabel", font=base)
    style.configure("Heading.TLabel", font=("Segoe UI", 10, "bold"))
    style.configure("Sentence.TLabel", font=("Segoe UI", 13))
    style.configure("Muted.TLabel", foreground="#6b6b6b", font=("Segoe UI", 9))
    style.configure("Metric.TLabel", font=("Segoe UI", 13, "bold"))
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    style.configure("Treeview", rowheight=int(22 * scale(root)), font=base)
    style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))


def main(db_path: Path = db.DB_FILE) -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    conn = db.connect(db_path)
    try:
        imported = db.ensure_imported(conn)
    except db.DatasetError as err:
        imported = 0
        messagebox.showerror("Import failed", str(err))
    root = tk.Tk()
    apply_style(root)
    app = AnnotatorApp(root, conn)
    if imported:
        app.status_var.set(f"Imported {imported} records from {db.CSV_FILE.name}. " + app.status_var.get())
    root.mainloop()
    conn.close()


if __name__ == "__main__":
    main()
