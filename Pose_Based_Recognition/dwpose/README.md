**What are the confidence values in my DWPose implementation?**

I'm using DWPose for extracting human body keypoints from Kathakali videos.

For every frame, DWPose detects 50 selected body joints. Along with the X and Y coordinates, it provides a **confidence score for each detected joint**.

**What does confidence mean?**

- X: Horizontal position of a joint.
- Y: Vertical position of a joint.
- Confidence: How confident DWPose is about that joint's detection/localization.

For example:
- 0.98 → Very high confidence in the detected joint.
- 0.75 → Relatively high confidence.
- 0.30 → Lower confidence; the joint detection may be uncertain.

These are pose-estimation confidence scores, NOT the probability that a Kathakali gesture belongs to a particular class.

**How I store them:**

Each processed video is saved as an `.npz` file containing:

- `keypoints_normalized`: `(100, 50, 2)` — normalized X/Y coordinates.
- `scores`: `(100, 50)` — DWPose confidence scores.
- `features`: `(100, 50, 3)` — X, Y and confidence combined.

So, my ST-GCN model can use confidence as an additional feature alongside joint coordinates to learn from the reliability of detected joints.

In short, confidence tells us how reliable each detected joint position is according to DWPose.