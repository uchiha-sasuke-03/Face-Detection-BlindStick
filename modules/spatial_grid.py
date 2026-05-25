def analyze_position(xyxy, frame_width, frame_height):
    """
    Analyzes the YOLO bounding box to determine its grid position and distance.
    xyxy: [x_min, y_min, x_max, y_max] from YOLO
    """
    x_min, y_min, x_max, y_max = xyxy
    
    # Calculate center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Determine horizontal position
    if center_x < frame_width / 3:
        h_pos = "left"
    elif center_x < (2 * frame_width) / 3:
        h_pos = "ahead"
    else:
        h_pos = "right"
        
    # Determine vertical position
    if center_y < frame_height / 3:
        v_pos = "top"
    elif center_y < (2 * frame_height) / 3:
        v_pos = "center"
    else:
        v_pos = "bottom"

    # Calculate distance proxy based on area
    box_area = (x_max - x_min) * (y_max - y_min)
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area
    
    if area_ratio > 0.05:
        distance = "close"
    elif area_ratio > 0.015:
        distance = "medium distance"
    else:
        distance = "far"
        
    # Simplify output for speech
    if h_pos == "ahead" and v_pos == "center":
        location = "straight ahead"
    else:
        location = f"{v_pos} {h_pos}"
        
    return location, distance
