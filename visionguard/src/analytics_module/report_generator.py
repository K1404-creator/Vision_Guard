"""
report_generator.py
--------------------
Consumes events stored by EventLogger/DBManager and produces:
  1. A summary dict (counts per event type, most-seen persons, etc.)
  2. Matplotlib chart images saved to REPORT_OUTPUT_DIR, giving the
     'Reporting / Analytics' functional module a visible output.
"""

import os
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display server needed
import matplotlib.pyplot as plt

from src.database.db_manager import DBManager
from src.utils.config import REPORT_OUTPUT_DIR
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, db: DBManager = None):
        self.db = db or DBManager()

    def summarize(self) -> dict:
        events = self.db.fetch_events(limit=10000)
        type_counts = Counter(e["event_type"] for e in events)
        person_counts = Counter(
            e["label"] for e in events if e["event_type"] == "face_recognized"
        )
        return {
            "total_events": len(events),
            "by_type": dict(type_counts),
            "top_persons": person_counts.most_common(5),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def plot_event_distribution(self, filename: str = "event_distribution.png") -> str:
        summary = self.summarize()
        by_type = summary["by_type"] or {"no_data": 0}

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(by_type.keys(), by_type.values(), color="#4C72B0")
        ax.set_title("Events Logged by Type")
        ax.set_ylabel("Count")
        ax.set_xlabel("Event Type")
        plt.xticks(rotation=20)
        plt.tight_layout()

        out_path = os.path.join(REPORT_OUTPUT_DIR, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved event distribution chart to %s", out_path)
        return out_path

    def plot_top_persons(self, filename: str = "top_persons.png") -> str:
        summary = self.summarize()
        top = summary["top_persons"] or [("no_data", 0)]
        names, counts = zip(*top)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(names, counts, color="#55A868")
        ax.set_title("Most Frequently Recognized Persons")
        ax.set_xlabel("Recognitions")
        plt.tight_layout()

        out_path = os.path.join(REPORT_OUTPUT_DIR, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved top-persons chart to %s", out_path)
        return out_path

    def generate_all(self) -> dict:
        """Convenience method used by main.py / CLI to build the full report bundle."""
        return {
            "summary": self.summarize(),
            "event_distribution_chart": self.plot_event_distribution(),
            "top_persons_chart": self.plot_top_persons(),
        }


if __name__ == "__main__":
    rg = ReportGenerator()
    result = rg.generate_all()
    print(result["summary"])
