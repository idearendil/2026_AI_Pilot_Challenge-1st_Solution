"""PyVista-free log parsing and tactical math for web playback."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEA_SIZE_M = 20_000.0
AIRCRAFT_MIN_DISPLAY_LENGTH_M = 45.0
AIRCRAFT_MAX_DISPLAY_LENGTH_M = 160.0
FEET_TO_M = 0.3048
DEFAULT_SPEED = 5.0
TRAIL_SECONDS = 10.0
DEFAULT_WEZ_MIN_RANGE_M = 500.0 * FEET_TO_M
DEFAULT_WEZ_RANGE_M = 3_000.0 * FEET_TO_M
DEFAULT_WEZ_ANGLE_DEG = 2.0

REQUIRED_COLUMNS = {
    "Time",
    "Longitude",
    "Latitude",
    "Altitude",
    "Roll (deg)",
    "Pitch (deg)",
    "Yaw (deg)",
}


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class AircraftTrack:
    """Parsed aircraft state history in local ENU display coordinates."""

    time: list[float]
    position: list[Vector3]
    roll_deg: list[float]
    pitch_deg: list[float]
    yaw_deg: list[float]
    health: list[float]


@dataclass(frozen=True)
class ViewerData:
    """Resolved log inputs and derived viewer state."""

    ownship_log: Path
    target_log: Path
    metadata_path: Path | None
    end_condition: str
    ownship: AircraftTrack
    target: AircraftTrack


def normalize_log_pair(
    selected_log: Path,
    paired_log: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve selected Blue/Red CSVs into ownship/target paths."""
    selected_log = Path(selected_log)
    if selected_log.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV log file: {selected_log}")

    if paired_log is not None:
        paired_log = Path(paired_log)
        ownship_log = selected_log if "[Blue]" in selected_log.name else paired_log
        target_log = selected_log if "[Red]" in selected_log.name else paired_log
        return ownship_log, target_log

    selected_name = selected_log.name
    if "[Blue]" in selected_name:
        ownship_log = selected_log
        target_name = selected_name.replace("ownship", "target").replace(
            "[Blue]", "[Red]"
        )
        target_log = selected_log.with_name(target_name)
    elif "[Red]" in selected_name:
        target_log = selected_log
        ownship_name = selected_name.replace("target", "ownship").replace(
            "[Red]", "[Blue]"
        )
        ownship_log = selected_log.with_name(ownship_name)
    else:
        raise ValueError(
            "Cannot infer Blue/Red pair from filename without [Blue]/[Red]: "
            f"{selected_log}"
        )

    if not ownship_log.exists():
        raise FileNotFoundError(f"Blue log not found: {ownship_log}")
    if not target_log.exists():
        raise FileNotFoundError(f"Red log not found: {target_log}")
    return ownship_log, target_log


def discover_log_pairs(logdir: Path) -> list[tuple[Path, Path]]:
    """Discover ownship/target CSV pairs under a log directory."""
    if not logdir.exists():
        return []

    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    for path in sorted(logdir.rglob("*.csv")):
        if "[Blue]" not in path.name and "[Red]" not in path.name:
            continue
        try:
            ownship, target = normalize_log_pair(path)
        except (FileNotFoundError, ValueError):
            continue
        key = (ownship.resolve(), target.resolve())
        if key not in seen:
            seen.add(key)
            pairs.append((ownship, target))
    return pairs


def read_log_rows(path: Path) -> list[dict[str, float]]:
    """Read a Tacview-style CSV log and return numeric rows."""
    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            parsed = {key: float(row[key]) for key in REQUIRED_COLUMNS}
            health_value = row.get("Health", row.get("health", "nan"))
            parsed["Health"] = parse_optional_float(health_value)
            rows.append(parsed)

    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def parse_optional_float(value: object) -> float:
    """Parse optional numeric CSV values as float, returning NaN on blanks."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_track(
    rows: list[dict[str, float]],
    ref_lat: float,
    ref_lon: float,
) -> AircraftTrack:
    """Convert geodetic rows to a local flat ENU frame."""
    position: list[Vector3] = []
    for row in rows:
        east, north = geodetic_to_local_m(
            lat_deg=row["Latitude"],
            lon_deg=row["Longitude"],
            ref_lat_deg=ref_lat,
            ref_lon_deg=ref_lon,
        )
        position.append((east, north, row["Altitude"]))

    return AircraftTrack(
        time=[row["Time"] for row in rows],
        position=position,
        roll_deg=[row["Roll (deg)"] for row in rows],
        pitch_deg=[row["Pitch (deg)"] for row in rows],
        yaw_deg=[row["Yaw (deg)"] for row in rows],
        health=[row["Health"] for row in rows],
    )


def geodetic_to_local_m(
    lat_deg: float,
    lon_deg: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
) -> tuple[float, float]:
    """Approximate WGS84 geodetic deltas as local meters."""
    earth_radius_m = 6_378_137.0
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    ref_lat_rad = math.radians(ref_lat_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    north = (lat_rad - ref_lat_rad) * earth_radius_m
    east = (lon_rad - ref_lon_rad) * earth_radius_m * math.cos(ref_lat_rad)
    return east, north


def build_viewer_data(
    ownship_log: Path,
    target_log: Path,
    metadata_path: Path | None,
    fallback_end_condition: str = "n/a",
) -> ViewerData:
    """Load logs and build a viewer-ready dataset."""
    own_rows = read_log_rows(ownship_log)
    target_rows = read_log_rows(target_log)
    ref_lat, ref_lon = first_row_ref(own_rows)
    ownship = build_track(own_rows, ref_lat, ref_lon)
    target = build_track(target_rows, ref_lat, ref_lon)
    end_condition = load_end_condition(metadata_path, fallback_end_condition)
    return ViewerData(
        ownship_log=Path(ownship_log),
        target_log=Path(target_log),
        metadata_path=metadata_path,
        end_condition=end_condition,
        ownship=ownship,
        target=target,
    )


def first_row_ref(rows: Iterable[dict[str, float]]) -> tuple[float, float]:
    """Return latitude/longitude from the first parsed CSV row."""
    first = next(iter(rows))
    return first["Latitude"], first["Longitude"]


def infer_summary_path(ownship_log: Path) -> Path | None:
    """Infer the summary JSON path generated beside an ownship CSV log."""
    marker = "_ownship_(F-16)[Blue].csv"
    name = Path(ownship_log).name
    if marker not in name:
        return None
    candidate = Path(ownship_log).with_name(name.replace(marker, "_summary.json"))
    return candidate if candidate.exists() else None


def load_metadata(metadata_path: Path | None) -> dict:
    """Load optional replay metadata JSON."""
    if metadata_path is None:
        return {}
    try:
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def load_end_condition(metadata_path: Path | None, fallback: str) -> str:
    """Load end condition text from metadata JSON when available."""
    metadata = load_metadata(metadata_path)
    if not metadata:
        return fallback
    end_condition = metadata.get("end_condition") or fallback
    outcome = metadata.get("outcome")
    if outcome:
        return f"{end_condition} ({outcome})"
    return str(end_condition)


def scene_extent_m(ownship: AircraftTrack, target: AircraftTrack) -> float:
    """Return the largest span needed to frame both tracks."""
    points = ownship.position + target.position
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    xy_span = max(max(xs) - min(xs), max(ys) - min(ys))
    z_span = max(zs) - min(zs)
    return max(xy_span, z_span, 1.0)


def sea_size_for_tracks(ownship: AircraftTrack, target: AircraftTrack) -> float:
    """Pick a sea plane size that covers the track extents with margin."""
    points = ownship.position + target.position
    horizontal_extent = max(
        max(abs(point[0]), abs(point[1])) for point in points
    ) * 2.0
    return max(SEA_SIZE_M, horizontal_extent + 2_000.0)


def aircraft_display_length_for_extent(extent_m: float) -> float:
    """Scale aircraft display size to the replay frame with conservative clamps."""
    scaled = max(extent_m * 0.012, AIRCRAFT_MIN_DISPLAY_LENGTH_M)
    return min(scaled, AIRCRAFT_MAX_DISPLAY_LENGTH_M)


def attitude_matrix(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[Vector3, Vector3, Vector3]:
    """Build a body-to-display rotation matrix for X-forward aircraft meshes."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(90.0 - yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    rot_y = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rot_z = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return matmul3(matmul3(rot_z, rot_y), rot_x)


def matmul3(
    first: tuple[Vector3, Vector3, Vector3],
    second: tuple[Vector3, Vector3, Vector3],
) -> tuple[Vector3, Vector3, Vector3]:
    """Multiply two 3x3 matrices."""
    rows: list[Vector3] = []
    for row in range(3):
        values: list[float] = []
        for col in range(3):
            values.append(sum(first[row][k] * second[k][col] for k in range(3)))
        rows.append((values[0], values[1], values[2]))
    return (rows[0], rows[1], rows[2])


def forward_vector(yaw_deg: float, pitch_deg: float) -> Vector3:
    """Compute aircraft forward vector in display coordinates."""
    matrix = attitude_matrix(0.0, pitch_deg, yaw_deg)
    direction = (
        matrix[0][0],
        matrix[1][0],
        matrix[2][0],
    )
    norm = vector_norm(direction)
    return vector_scale(direction, 1.0 / norm) if norm > 0.0 else (1.0, 0.0, 0.0)


def nearest_index(track: AircraftTrack, sim_time: float) -> int:
    """Return the nearest prior sample index for replay time."""
    index = bisect_right(track.time, sim_time) - 1
    return max(0, min(index, len(track.time) - 1))


def trail_points(
    track: AircraftTrack,
    sim_time: float,
    trail_seconds: float,
) -> list[Vector3]:
    """Return display positions inside the trailing time window."""
    start_time = sim_time - trail_seconds
    start = bisect_left(track.time, start_time)
    end = bisect_right(track.time, sim_time)
    return track.position[start:end]


def speed_at(track: AircraftTrack, index: int) -> float:
    """Estimate speed from neighboring log samples in meters per second."""
    if len(track.time) < 2:
        return 0.0
    prev_i = max(0, index - 1)
    next_i = min(len(track.time) - 1, index + 1)
    dt = track.time[next_i] - track.time[prev_i]
    if dt <= 0.0:
        return 0.0
    return vector_norm(vector_sub(track.position[next_i], track.position[prev_i])) / dt


def velocity_at(track: AircraftTrack, index: int) -> Vector3:
    """Estimate velocity vector from neighboring log samples."""
    if len(track.time) < 2:
        return (0.0, 0.0, 0.0)
    prev_i = max(0, index - 1)
    next_i = min(len(track.time) - 1, index + 1)
    dt = track.time[next_i] - track.time[prev_i]
    if dt <= 0.0:
        return (0.0, 0.0, 0.0)
    return vector_scale(vector_sub(track.position[next_i], track.position[prev_i]), 1.0 / dt)


def angle_between_deg(first: Vector3, second: Vector3) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    first_norm = vector_norm(first)
    second_norm = vector_norm(second)
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 0.0
    cosine = vector_dot(first, second) / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def in_wez(
    range_m: float,
    ata_deg: float,
    min_range_m: float = DEFAULT_WEZ_MIN_RANGE_M,
    max_range_m: float = DEFAULT_WEZ_RANGE_M,
    full_angle_deg: float = DEFAULT_WEZ_ANGLE_DEG,
) -> bool:
    """Return whether target geometry is inside the simplified WEZ frustum."""
    return (
        min_range_m <= range_m <= max_range_m
        and ata_deg <= max(0.0, full_angle_deg / 2.0)
    )


def tactical_snapshot(
    ownship: AircraftTrack,
    target: AircraftTrack,
    sim_time: float,
    wez_min_range: float = DEFAULT_WEZ_MIN_RANGE_M,
    wez_range: float = DEFAULT_WEZ_RANGE_M,
    wez_angle: float = DEFAULT_WEZ_ANGLE_DEG,
) -> dict[str, float | bool]:
    """Compute key HUD values for one replay time."""
    own_i = nearest_index(ownship, sim_time)
    target_i = nearest_index(target, sim_time)
    own_pos = ownship.position[own_i]
    target_pos = target.position[target_i]
    relative = vector_sub(target_pos, own_pos)
    distance = vector_norm(relative)
    own_forward = forward_vector(ownship.yaw_deg[own_i], ownship.pitch_deg[own_i])
    target_forward = forward_vector(target.yaw_deg[target_i], target.pitch_deg[target_i])
    own_ata = angle_between_deg(own_forward, relative)
    target_ata = angle_between_deg(target_forward, vector_scale(relative, -1.0))
    own_aa = angle_between_deg(target_forward, vector_scale(relative, -1.0))
    relative_velocity = vector_sub(velocity_at(target, target_i), velocity_at(ownship, own_i))
    closure = -vector_dot(relative, relative_velocity) / distance if distance > 0.0 else 0.0
    return {
        "own_index": own_i,
        "target_index": target_i,
        "distance_m": distance,
        "closure_mps": closure,
        "relative_alt_m": target_pos[2] - own_pos[2],
        "own_ata_deg": own_ata,
        "target_ata_deg": target_ata,
        "own_aa_deg": own_aa,
        "own_speed_mps": speed_at(ownship, own_i),
        "target_speed_mps": speed_at(target, target_i),
        "own_wez": in_wez(distance, own_ata, wez_min_range, wez_range, wez_angle),
        "target_wez": in_wez(distance, target_ata, wez_min_range, wez_range, wez_angle),
    }


def vector_sub(first: Vector3, second: Vector3) -> Vector3:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def vector_scale(vector: Vector3, scale: float) -> Vector3:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def vector_dot(first: Vector3, second: Vector3) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def vector_norm(vector: Vector3) -> float:
    return math.sqrt(vector_dot(vector, vector))


def track_to_json(track: AircraftTrack) -> dict[str, list]:
    """Serialize an AircraftTrack to browser-friendly arrays."""
    return {
        "time": track.time,
        "position": track.position,
        "rollDeg": track.roll_deg,
        "pitchDeg": track.pitch_deg,
        "yawDeg": track.yaw_deg,
        "health": [None if math.isnan(value) else value for value in track.health],
    }


def viewer_data_to_json(data: ViewerData) -> dict:
    """Serialize loaded logs and derived display constants."""
    extent = scene_extent_m(data.ownship, data.target)
    start_time = max(data.ownship.time[0], data.target.time[0])
    end_time = min(data.ownship.time[-1], data.target.time[-1])
    metadata = load_metadata(data.metadata_path)
    initial = tactical_snapshot(data.ownship, data.target, start_time)
    return {
        "logs": {
            "ownship": data.ownship_log.name,
            "target": data.target_log.name,
            "metadata": data.metadata_path.name if data.metadata_path else None,
        },
        "metadata": metadata,
        "endCondition": data.end_condition,
        "startTime": start_time,
        "endTime": end_time,
        "duration": max(0.0, end_time - start_time),
        "sceneExtentM": extent,
        "seaSizeM": sea_size_for_tracks(data.ownship, data.target),
        "aircraftDisplayLengthM": aircraft_display_length_for_extent(extent),
        "defaults": {
            "speed": DEFAULT_SPEED,
            "trailSeconds": TRAIL_SECONDS,
            "wezMinRangeM": DEFAULT_WEZ_MIN_RANGE_M,
            "wezRangeM": DEFAULT_WEZ_RANGE_M,
            "wezAngleDeg": DEFAULT_WEZ_ANGLE_DEG,
        },
        "initialTactical": initial,
        "ownship": track_to_json(data.ownship),
        "target": track_to_json(data.target),
    }


def parse_obj_mesh(path: Path) -> dict:
    """Parse a small OBJ mesh into normalized vertices and triangle faces."""
    vertices: list[Vector3] = []
    triangles: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                face = [_obj_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(face) - 1):
                    triangles.append((face[0], face[i], face[i + 1]))

    if not vertices or not triangles:
        raise ValueError(f"Invalid OBJ mesh: {path}")

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    center = (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    )
    length = max(xs) - min(xs)
    if length <= 0.0:
        raise ValueError(f"Invalid mesh length: {path}")

    normalized = [
        (
            (v[0] - center[0]) / length,
            (v[1] - center[1]) / length,
            (v[2] - center[2]) / length,
        )
        for v in vertices
    ]
    return {
        "vertices": normalized,
        "triangles": triangles,
        "source": Path(path).name,
        "unitLengthAxis": "x",
    }


def _obj_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/")[0])
    return raw - 1 if raw > 0 else vertex_count + raw
