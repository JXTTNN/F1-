# F1OPT Quick Start

## Step 1: Download

Get the latest binary from [Releases](https://github.com/JXTTNN/F1-/releases). No Python or dependencies needed.

```bash
./f1opt --help
```

## Step 2: Driver Feedback

```bash
# See available examples
./f1opt feedback --list-examples

# Corner-level
./f1opt feedback --track suzuka --question "Why am I understeering into T1?"

# Sector-level
./f1opt feedback --track suzuka --question "S2 feels slow through the esses"

# Overall
./f1opt feedback --track bahrain --question "How much time can I still find?"
```

**Precision levels:**

| Level | Scope | Example |
|---|---|---|
| **corner** | Specific corner | "T1 understeer" / "T130R exit slow" |
| **sector** | Sector / section | "S2 directional issues" / "S3 unstable at speed" |
| **overall** | Whole lap | "How much faster?" / "Tire temps uneven" |

## Step 3: Setup Search

```bash
# Differential evolution (100 rounds)
./f1opt search --track suzuka --iterations 100

# Bayesian optimization
./f1opt bayesian --track monza --iterations 15 --acquisition ei

# Predict lap time
./f1opt predict --track suzuka --setup-json '{"front_wing":30,"rear_wing":25,"on_throttle_diff":80,"off_throttle_diff":50,"front_suspension":5,"rear_suspension":5,"front_arb":5,"rear_arb":5,"front_tyre_pressure":23,"rear_tyre_pressure":21,"front_camber":-3.0,"rear_camber":-1.5,"front_toe":0.05,"rear_toe":0.15,"front_ride_height":5,"rear_ride_height":5,"front_brake_bias":50,"brake_pressure":90,"fuel_load":30}'
```

## More

| What | Where |
|---|---|
| Full docs | [README.md](README.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Examples | [examples/](examples/) |
| API docs | Run `./f1opt serve` then open http://127.0.0.1:8000/docs |
| Source | [f1opt/](f1opt/) |
| Tests | [tests/](tests/) |