class RuleBasedClassifier:
    def __init__(self):
        # COCO dataset keypoint indices for YOLOv8 pose
        self.L_SHOULDER = 5
        self.R_SHOULDER = 6
        self.L_WRIST = 9
        self.R_WRIST = 10
        self.L_HIP = 11
        self.R_HIP = 12

    def classify_shot(self, keypoints):
        """
        Takes a single player's keypoints and returns a shot type string.
        Keypoints shape expected: (17, 2) [x, y] coordinates.
        """
        if keypoints is None or len(keypoints) < 17:
            return "Idle"

        r_shoulder = keypoints[self.R_SHOULDER]
        l_shoulder = keypoints[self.L_SHOULDER]
        r_wrist = keypoints[self.R_WRIST]
        r_hip = keypoints[self.R_HIP]

        # If keypoints are [0, 0], it means the model couldn't see that body part
        if r_wrist[0] == 0 or r_hip[0] == 0 or r_shoulder[0] == 0:
            return "Idle"

        # Check if arm is raised (y-axis is inverted: lower value = higher up)
        if r_wrist[1] > r_hip[1]:
            return "Idle" 

        # Classify based on x-coordinates (assuming right-handed players for prototype)
        if r_wrist[0] > r_shoulder[0]:
            return "Forehand"
        elif r_wrist[0] < l_shoulder[0]:
            return "Backhand"
        else:
            return "Preparing"