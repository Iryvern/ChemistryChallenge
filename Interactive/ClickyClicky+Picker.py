import pygame

# Initialize Pygame
pygame.init()

# Set up the window
window_width = 729
window_height = 696
window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
pygame.display.set_caption("Click to color white areas")

# Set up the clock
clock = pygame.time.Clock()

# Load the background image
background_image = pygame.image.load("inverted_combined_mask.png").convert()

# Scale the background image to fit the screen
background_image = pygame.transform.scale(background_image, (window_width, window_height))

def change_pixel_color(surface, pos, new_color):    
    flood_fill(surface, pos, new_color, tolerance=10)
    pygame.display.update()

def flood_fill(surface, start_pos, new_color, tolerance=0):
    # Get the starting color and convert it to a tuple if necessary
    start_color = surface.get_at(start_pos)[:3]
    if not isinstance(start_color, tuple):
        start_color = tuple(start_color)
    
    # Create a set to keep track of visited pixels
    visited = set()
    
    # Create a stack to store the pixels to be checked
    stack = [start_pos]
    
    # Loop until the stack is empty
    while stack:
        # Pop the next pixel from the stack
        pos = stack.pop()
        
        # Check if this pixel has already been visited
        if pos in visited:
            continue
        
        # Get the color of the current pixel and check if it matches the starting color
        color = surface.get_at(pos)[:3]
        if abs(color[0] - start_color[0]) > tolerance or \
           abs(color[1] - start_color[1]) > tolerance or \
           abs(color[2] - start_color[2]) > tolerance:
            continue
        
        # Change the color of the current pixel and mark it as visited
        surface.set_at(pos, new_color)
        visited.add(pos)
        
        # Add the neighboring pixels to the stack
        if pos[0] > 0:
            stack.append((pos[0]-1, pos[1]))
        if pos[0] < surface.get_width()-1:
            stack.append((pos[0]+1, pos[1]))
        if pos[1] > 0:
            stack.append((pos[0], pos[1]-1))
        if pos[1] < surface.get_height()-1:
            stack.append((pos[0], pos[1]+1))

class Button:
    def __init__(self, x, y, w, h, text, font, font_color, bg_color, action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.font_color = font_color
        self.bg_color = bg_color
        self.action = action

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        text_surface = self.font.render(self.text, True, self.font_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()

class ColorPicker:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.image = pygame.Surface((w, h))
        self.image.fill((255, 255, 255))
        self.rad = h//2
        self.pwidth = w-self.rad*2
        for i in range(self.pwidth):
            color = pygame.Color(0)
            color.hsla = (int(360*i/self.pwidth), 100, 50, 100)
            pygame.draw.rect(self.image, color, (i+self.rad, h//3, 1, h-2*h//3))
        self.p = 0

    def get_color(self):
        color = pygame.Color(0)
        color.hsla = (int(self.p * self.pwidth*2), 100, 50, 100)
        return color

    def update(self):
        moude_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        if moude_buttons[0] and self.rect.collidepoint(mouse_pos):
            self.p = (mouse_pos[0] - self.rect.left - self.rad) / self.pwidth
            self.p = (max(0, min(self.p, 1)))

    def draw(self, surf):
        surf.blit(self.image, self.rect)
        center = self.rect.left + self.rad + self.p * self.pwidth, self.rect.centery
        pygame.draw.circle(surf, self.get_color(), center, self.rect.height // 2)

cp = ColorPicker(5, 60, 200, 30)

def button_action():
    pygame.image.save(background_image, "saved_image.png")
    print("Saved image")

button_font = pygame.font.Font(None, 36)
button_text = "Save"
button_color = (0, 255, 0)
button = Button(10, 10, 100, 50, button_text, button_font, (0, 0, 0), button_color, lambda: button_action() )

# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the mouse position and draw a circle there
            mouse_pos = pygame.mouse.get_pos()
            # If the click was not inside the ColorPicker or the Button, perform the color change
            if not cp.rect.collidepoint(mouse_pos) and not button.rect.collidepoint(mouse_pos):
                # Get the color of the pixel at the mouse position
                color = pygame.display.get_surface().get_at(mouse_pos)[:3]
                new_color = cp.get_color()  # get the color from the color picker
                change_pixel_color(background_image, mouse_pos, new_color)
                print(f"Colored at ({mouse_pos[0]}, {mouse_pos[1]}) - Color: {color} -> {new_color}")
                        
        elif event.type == pygame.VIDEORESIZE:
            # Resize the window and scale the background image to fit
            window_width = event.w
            window_height = event.h
            window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            background_image = pygame.transform.scale(background_image, (window_width, window_height))

        button.handle_event(event)
    
    # Draw the background image
    window.blit(background_image, (0, 0))

    # Draw the button
    button.draw(window)

    cp.update()
    
    cp.draw(window)

    # Update the screen
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
