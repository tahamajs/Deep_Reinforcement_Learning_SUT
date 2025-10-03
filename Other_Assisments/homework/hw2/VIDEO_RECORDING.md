# Video Recording for Policy Gradient Training

## Overview

This package now includes automatic video recording capabilities to visualize agent performance before and after training.

## Features

- **Before Training Videos**: Records episodes with untrained (random-like) policy at iteration 0
- **After Training Videos**: Records episodes with fully trained policy after all iterations
- **Automatic Organization**: Videos saved in structured directories by environment and configuration
- **Performance Metrics**: Displays average returns and improvement

## Usage

### Automatic Recording (via run_all_hw2.sh)

The main training script now automatically records videos:

```bash
./run_all_hw2.sh
```

Videos will be saved to `results_hw2/videos/<env>/<config>/before|after/`

### Manual Recording

To record videos for a specific trained agent:

```bash
python record_videos.py --logdir results_hw2/logs/CartPole-v0_Vanilla_... --env CartPole-v0
```

### Command-line Options

When running `run_pg.py` directly:

```bash
python run_pg.py CartPole-v0 --exp_name test --record_video
```

**Flags:**
- `--record_video`: Enable before/after video recording (default: disabled)

## Video Structure

```
results_hw2/
└── videos/
    ├── CartPole-v0/
    │   ├── Vanilla/
    │   │   ├── before/
    │   │   │   └── episode*.mp4
    │   │   └── after/
    │   │       └── episode*.mp4
    │   ├── RTG/
    │   │   ├── before/
    │   │   └── after/
    │   └── ...
    ├── LunarLander-v2/
    │   └── ...
    └── HalfCheetah-v2/
        └── ...
```

## Dependencies

Video recording requires:
- `ffmpeg` (for video encoding)
- `gym` with Monitor wrapper
- Rendering support for your environment

### Installing ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Viewing Videos

Videos are saved in MP4 format and can be viewed with any standard video player:

```bash
# Open videos directly
open results_hw2/videos/CartPole-v0/Vanilla/after/episode-*.mp4

# Or browse the directory
cd results_hw2/videos
```

## Implementation Details

### src/video_recorder.py

Core video recording utilities:
- `record_episode_video()`: Records episodes and saves as video
- `record_before_after_videos()`: Handles before/after comparison
- Automatic directory management

### Integration Points

1. **run_pg.py**: Integrated at training start and end
2. **run_all_hw2.sh**: Automatically enabled for all experiments
3. **record_videos.py**: Standalone script for post-hoc recording

## Troubleshooting

### "No such file or directory: 'ffmpeg'"

Install ffmpeg using the commands above.

### Videos not rendering

Some environments don't support rendering. CartPole-v0 and LunarLander-v2 work well. For HalfCheetah-v2, you may need additional MuJoCo setup.

### Out of memory

Reduce `num_episodes` parameter when recording:

```python
record_before_after_videos(..., num_episodes=1)
```

## Performance Impact

- Recording adds ~10-30 seconds per configuration
- Videos are recorded at standard frame rates
- Minimal impact on training performance (recorded after training)

## Examples

Check the saved videos to see:
- Initial random behavior (before)
- Learned optimal behavior (after)
- Visual comparison of different configurations (Vanilla vs RTG vs Baseline)

Enjoy visualizing your trained agents! 🎬
