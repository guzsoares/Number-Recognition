import pygame
from PIL import Image
from inference import load_model, predict
import torchvision.transforms as transforms

pygame.init()

WIDTH, HEIGHT = 280, 280 
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Desenhe um número (0-9)")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

font = pygame.font.SysFont("Arial", 28)

drawing = False
model = load_model() 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

prediction = None

def clear():
    WIN.fill(BLACK)
    pygame.display.update()

clear()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                prediction = None
                clear()
            elif event.key == pygame.K_SPACE:
                pygame.image.save(WIN, "temp.png")
                img = Image.open("temp.png").convert("L")
                img = transform(img)
                pred = predict(model, img)
                prediction = pred

    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(WIN, WHITE, mouse_pos, 10)

    if prediction is not None:
        text = font.render(f"Previsão: {prediction}", True, WHITE)
        WIN.blit(text, (10, 10))

    pygame.display.update()

pygame.quit()