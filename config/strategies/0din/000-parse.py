import argparse
import json
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def msg(text: str, log_level: int, label: str = "", color: int = 1) -> None:
    """Show a in the console.

    Parameters
    ----------
    text : str
        The message to show.
    log_level: int
        Logging level. See Python's logging module's documentation for details.
    label : str, default is ""
        An optional label to prepend to the message.
    color : int, default is 1
        The color code to use for the label, if present. The default value is 1 (bold).
    """
    if label:
        label += ": "

    logger.log(level=log_level, msg=text, extra={"label": label, "color": color})


def info(text: str, color: int = 1) -> None:
    """Show an informative message in the console."""
    msg(text, log_level=logging.INFO, label="INFO", color=color)


def warn(text: str, color: int = 91) -> None:
    """Show a warning message in the console."""
    msg(text, log_level=logging.WARNING, label="WARN", color=color)


def debug(text: str, color: int = 91) -> None:
    """Show a debugging message in the console."""
    msg(text, log_level=logging.DEBUG, label="DEBUG", color=color)


def fatal(text: str, color: int = 91) -> None:
    """Show a fatal error message in the console and exit the program."""
    msg(text, log_level=logging.CRITICAL, label="FATAL", color=color)

    exit(1)


def parse(record: dict) -> None:
    if "children" in record:
        for child in record["children"]:
            parse(child)

    if "kind" in record and record["kind"] == "technique":
        name = record["name"]
        desc = record["description"]
        file = Path(name.lower().replace(" ", "_") + ".txt")

        debug(f"name: {name} => {desc}")

        if file.exists():
            warn(f"{file} already exists. It will not overwritten")
            return

        with open(file, "w") as f:
            f.write(f"## {name}\n")
            f.write(desc)


def main(args: argparse.Namespace) -> None:
    with open(args.path, "r") as f:
        records = json.load(f)

        for record in records:
            parse(record)

    info("DONE!")


class ColoredFormatter(logging.Formatter):
    def format(self, record):  # Provide defaults for missing fields to prevent KeyError
        if not hasattr(record, "color"):
            record.color = 1
        if not hasattr(record, "label"):
            record.label = ""

        return super().format(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse 0din's JSON taxonomy file from https://0din.ai/research/taxonomy",
    )
    parser.add_argument("path", help="Path to 0din's taxonomy file")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warn"],
        default="info",
        help="Set log level (default: %(default)s)",
    )

    args = parser.parse_args()

    if not Path(args.path).exists():
        fatal(f"{args.path} not found")

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(format="%(label)s%(message)s", level=log_level)

    main(args)
