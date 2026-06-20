#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --policy <name> --version <name> --destination-root <path>

Export a model from colosseum's registry to an arena checkout.
Updates the destination's models.yaml and per-policy models.yaml automatically.

Example:
  $(basename "$0") --policy t1-velocity --version v5 --destination-root ../arena
EOF
    exit 1
}

POLICY=""
VERSION=""
DEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy)            POLICY="$2";  shift 2 ;;
        --version)           VERSION="$2"; shift 2 ;;
        --destination-root)  DEST="$2";    shift 2 ;;
        -h|--help)           usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

[[ -n "$POLICY"  ]] || { echo "Missing --policy";  usage; }
[[ -n "$VERSION" ]] || { echo "Missing --version"; usage; }
[[ -n "$DEST"    ]] || { echo "Missing --destination-root"; usage; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGISTRY="$SCRIPT_DIR/../../../models/registry.yaml"
MODELS_ROOT="$SCRIPT_DIR/../../../models"

if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY"
    exit 1
fi

# ── Extract the ONNX file path from colosseum's registry.yaml ─────────────
#
# registry.yaml structure:
#   t1-velocity:
#     models:
#       v5:
#         file: v5/t1-velocity_ppo_v5.onnx
#     default: v0
#

FILE=$(awk -v policy="$POLICY" -v version="$VERSION" '
    BEGIN { in_policy = 0; in_models = 0; in_version = 0; file = "" }

    /^[a-zA-Z]/ {
        in_policy = ($0 ~ "^" policy ":")
        in_models = 0
        in_version = 0
    }

    in_policy && $0 ~ /^  models:/ { in_models = 1; in_version = 0; next }

    in_policy && in_models && $0 ~ /^    / && $0 ~ /:$/ {
        in_version = ($0 ~ "^    " version ":")
        next
    }

    in_policy && in_models && in_version && /^      file:/ {
        sub(/^[[:space:]]*file:[[:space:]]*/, "")
        gsub(/"/, "")
        sub(/[[:space:]]*$/, "")
        file = $0
        exit
    }

    END { if (file == "") exit 1; print file }
' "$REGISTRY")

if [[ -z "$FILE" ]]; then
    echo "Version '$VERSION' not found for policy '$POLICY' in registry.yaml"
    exit 1
fi

# ── Resolve absolute source path ─────────────────────────────────────────
# FILE is relative to <models>/<policy>/ (e.g. "v5/t1-velocity_ppo_v5.onnx")
SRC="$MODELS_ROOT/$POLICY/$FILE"
if [[ ! -f "$SRC" ]]; then
    echo "ONNX file not found: $SRC"
    exit 1
fi

# ── Copy model files to destination ──────────────────────────────────────
SRC_MODEL_DIR="$(dirname "$SRC")"
DST_MODEL_DIR="$DEST/models/$POLICY/$VERSION"
mkdir -p "$DST_MODEL_DIR"

# arena loads per-joint deploy gains from gains.yaml next to the ONNX (the Policy
# base constructor requires it). Always (re)generate from config.yaml so a stale
# gains.yaml is never shipped, then port it alongside the ONNX.
if [[ -f "$SRC_MODEL_DIR/config.yaml" ]]; then
    python -m colosseum.scripts.export_gains "$SRC_MODEL_DIR"
fi

cp -r "$SRC_MODEL_DIR"/*.onnx* "$DST_MODEL_DIR/"
if [[ -f "$SRC_MODEL_DIR/gains.yaml" ]]; then
    cp "$SRC_MODEL_DIR/gains.yaml" "$DST_MODEL_DIR/"
    echo "Copied *.onnx* + gains.yaml → $DST_MODEL_DIR/"
else
    echo "WARNING: no gains.yaml for $SRC_MODEL_DIR — arena deploy will fail to load gains"
    echo "Copied $SRC_MODEL_DIR/*.onnx* → $DST_MODEL_DIR/"
fi

# ── Update per-policy models.yaml ────────────────────────────────────────
POLICY_YAML="$DEST/models/$POLICY/models.yaml"

if [[ -f "$POLICY_YAML" ]]; then
    if grep -q "^  $VERSION:" "$POLICY_YAML"; then
        # Update the path in-place (algorithm name may have changed)
        awk -v ver="  $VERSION:" -v file="$FILE" '
            $0 == ver { in_v = 1; print; next }
            in_v && /^    path:/ { print "    path: " file; in_v = 0; next }
            { print }
        ' "$POLICY_YAML" > "${POLICY_YAML}.tmp" && mv "${POLICY_YAML}.tmp" "$POLICY_YAML"
        echo "Updated path for $VERSION in $POLICY_YAML"
    else
        cat >> "$POLICY_YAML" <<EOF
  $VERSION:
    path: $FILE
EOF
    fi
else
    mkdir -p "$(dirname "$POLICY_YAML")"
    cat > "$POLICY_YAML" <<EOF
default: $VERSION
versions:
  $VERSION:
    path: $FILE
EOF
    echo "Created $POLICY_YAML"
fi

# ── Update top-level models.yaml ─────────────────────────────────────────
TOP_YAML="$DEST/models/models.yaml"

if [[ -f "$TOP_YAML" ]]; then
    if grep -q "^$POLICY:" "$TOP_YAML"; then
        echo "Policy $POLICY already registered in $TOP_YAML (default not changed)"
    else
        cat >> "$TOP_YAML" <<EOF
$POLICY: $VERSION
EOF
        echo "Added $POLICY to $TOP_YAML"
    fi
else
    mkdir -p "$(dirname "$TOP_YAML")"
    cat > "$TOP_YAML" <<EOF
$POLICY: $VERSION
EOF
    echo "Created $TOP_YAML"
fi

echo "Done. $POLICY $VERSION exported to $DEST/models/"
