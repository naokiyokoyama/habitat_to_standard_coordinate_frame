import matplotlib.pyplot as plt
import numpy as np

import cv2
import numpy as np


def generate_text_image(width: int, text: str) -> np.ndarray:
    """
    Generates an image of the given text with line breaks, honoring given width.

    Args:
        width (int): Width of the image.
        text (str): Text to be drawn.

    Returns:
        np.ndarray: Text drawn on white image with the given width.
    """
    # Define the parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 10  # Spacing between lines in pixels

    # Calculate the maximum width and height of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    max_width = width - 20  # Allow some padding
    max_height = text_size[1] + line_spacing

    # Split the text into words
    words = text.split()

    # Initialize variables for text positioning
    x = 10
    y = text_size[1]

    to_draw = []

    # Iterate over the words and add them to the image
    num_rows = 1
    for word in words:
        # Get the size of the word
        word_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)

        # Check if adding the word exceeds the maximum width
        if x + word_size[0] > max_width:
            # Add a line break before the word
            y += max_height
            x = 10
            num_rows += 1

        # Draw the word on the image
        to_draw.append((word, x, y))

        # Update the position for the next word
        x += word_size[0] + 5  # Add some spacing between words

    # Create a blank white image with the calculated dimensions
    image = 255 * np.ones((max_height * num_rows, width, 3), dtype=np.uint8)
    for word, x, y in to_draw:
        cv2.putText(
            image,
            word,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def add_text_to_image(image: np.ndarray, text: str, top: bool = False) -> np.ndarray:
    """
    Adds text to the given image.

    Args:
        image (np.ndarray): Input image.
        text (str): Text to be added.
        top (bool, optional): Whether to add the text to the top or bottom of the image.

    Returns:
        np.ndarray: Image with text added.
    """
    width = image.shape[1]
    text_image = generate_text_image(width, text)
    if top:
        combined_image = np.vstack([text_image, image])
    else:
        combined_image = np.vstack([image, text_image])

    return combined_image


def visualize_point_cloud(
    points,
    point_size=50,
    point_color="b",
    alpha=0.6,
    title="3D Point Cloud",
    show_grid=True,
    axis_labels=("X", "Y", "Z"),
):
    """
    Visualize a list of 3D coordinates as a point cloud.

    Parameters:
    -----------
    points : list of tuples or numpy array
        List of (x, y, z) coordinates to plot
    point_size : int, optional
        Size of the points in the visualization
    point_color : str or list, optional
        Color(s) of the points
    alpha : float, optional
        Transparency of points (0 to 1)
    title : str, optional
        Title of the plot
    show_grid : bool, optional
        Whether to show grid lines
    axis_labels : tuple of str, optional
        Labels for the x, y, and z axes

    Returns:
    --------
    fig : matplotlib figure
        The generated figure object
    """
    # Convert points to numpy array if not already
    points = np.array(points)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Plot the points
    scatter = ax.scatter(x, y, z, c=point_color, s=point_size, alpha=alpha)

    # Customize the plot
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)

    # Toggle grid
    ax.grid(show_grid)

    # Add a color bar if colors vary
    if isinstance(point_color, (list, np.ndarray)):
        plt.colorbar(scatter)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return fig
