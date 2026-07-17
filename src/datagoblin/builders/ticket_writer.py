from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datagoblin.domain.builders import BuilderTicket
from datagoblin.serialization import dump_structured


def write_tickets(tickets: Iterable[BuilderTicket], output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ticket in tickets:
        path = output_directory / f"{ticket.ticket_id}.json"
        dump_structured(path, ticket.to_jsonable())
        paths.append(path)
    return tuple(paths)
