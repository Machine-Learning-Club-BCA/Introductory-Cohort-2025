import pygame
import random
import numpy as np

# Simple Geometry Dash-like environment + Q-learning agent

WIDTH, HEIGHT = 600, 200
PLAYER_X = 100
GRAVITY = 1
JUMP_STRENGTH = -12
FPS = 60
TOTAL_EPISODES = 1000

BACKGROUND = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BUTTON = (40, 40, 40)
BUTTON_HOVER = (70, 70, 70)

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_y = HEIGHT - 40
        self.vel_y = 0
        self.obstacle_x = WIDTH
        self.done = False
        self.score = 0
        return self.get_state()

    def get_state(self):
        return (self.player_y, self.vel_y, self.obstacle_x)

    def step(self, action):
        reward = 1

        # action: 0 = nothing, 1 = jump
        if action == 1 and self.player_y >= HEIGHT - 40:
            self.vel_y = JUMP_STRENGTH

        # physics
        self.vel_y += GRAVITY
        self.player_y += self.vel_y

        if self.player_y >= HEIGHT - 40:
            self.player_y = HEIGHT - 40
            self.vel_y = 0

        # move obstacle
        self.obstacle_x -= 5
        if self.obstacle_x < -20:
            self.obstacle_x = WIDTH + random.randint(0, 200)
            self.score += 1

        # collision
        if abs(self.obstacle_x - PLAYER_X) < 20 and self.player_y > HEIGHT - 60:
            reward = -100
            self.done = True

        return self.get_state(), reward, self.done


class QAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0

    def discretize(self, state):
        y, vy, ox = state
        return (int(y/20), int(vy/5), int(ox/50))

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return np.argmax(self.get_q(state))

    def learn(self, s, a, r, s2):
        q = self.get_q(s)
        q_next = self.get_q(s2)
        q[a] += self.alpha * (r + self.gamma * max(q_next) - q[a])


def draw_text(surface, font, text, color, x, y):
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


def draw_button(surface, rect, font, text, mouse_pos):
    color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, WHITE, rect, 2, border_radius=10)
    label = font.render(text, True, WHITE)
    label_rect = label.get_rect(center=rect.center)
    surface.blit(label, label_rect)


def render_game(surface, game, font, episode=None, total_reward=None, status_text=None):
    surface.fill(BACKGROUND)
    pygame.draw.rect(surface, GREEN, (PLAYER_X, game.player_y, 20, 20))
    pygame.draw.rect(surface, RED, (game.obstacle_x, HEIGHT - 40, 20, 40))
    if episode is not None:
        draw_text(surface, font, f"Episode: {episode}", WHITE, 10, 10)
    if total_reward is not None:
        draw_text(surface, font, f"Reward: {total_reward}", WHITE, 10, 30)
    if status_text:
        draw_text(surface, font, status_text, WHITE, 10, 50)
    pygame.display.flip()


def train_agent(agent, game, episodes, screen=None, clock=None, render=False):
    last_score = 0
    last_reward = 0
    info_font = pygame.font.SysFont("arial", 18)

    for episode in range(episodes):
        state = agent.discretize(game.reset())
        total_reward = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            action = agent.choose_action(state)
            next_state_raw, reward, done = game.step(action)
            next_state = agent.discretize(next_state_raw)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if render and screen is not None:
                render_game(
                    screen,
                    game,
                    font=info_font,
                    episode=episode + 1,
                    total_reward=total_reward,
                    status_text="Watching training...",
                )
                if clock is not None:
                    clock.tick(FPS)

            if done:
                break

        agent.epsilon *= 0.995
        last_score = game.score
        last_reward = total_reward

    return last_score, last_reward


def play_agent_showcase(agent, game, screen, clock):
    info_font = pygame.font.SysFont("arial", 18)
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    best_score = 0
    best_reward = 0

    try:
        state = agent.discretize(game.reset())
        total_reward = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            action = agent.choose_action(state)
            next_state_raw, reward, done = game.step(action)
            state = agent.discretize(next_state_raw)
            total_reward += reward

            render_game(
                screen,
                game,
                font=info_font,
                episode=1,
                total_reward=total_reward,
                status_text="Final agent showcase",
            )
            clock.tick(FPS)

            if game.score > best_score:
                best_score = game.score
                best_reward = total_reward

            if done:
                return game.score, total_reward

        return best_score, best_reward
    finally:
        agent.epsilon = old_epsilon


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Geometry Dash Trainer")

title_font = pygame.font.SysFont("arial", 28, bold=True)
body_font = pygame.font.SysFont("arial", 20)
small_font = pygame.font.SysFont("arial", 16)

agent = QAgent()
game = Game()

fast_track_button = pygame.Rect(WIDTH // 2 - 140, HEIGHT // 2 - 35, 280, 50)
watch_button = pygame.Rect(WIDTH // 2 - 140, HEIGHT // 2 + 30, 280, 50)


def show_menu():
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if fast_track_button.collidepoint(event.pos):
                    return "fast_track"
                if watch_button.collidepoint(event.pos):
                    return "watch"

        screen.fill(BACKGROUND)
        draw_text(screen, title_font, "Geometry Dash Trainer", WHITE, 140, 40)
        draw_text(screen, body_font, "Pick how you want to train:", WHITE, 180, 80)
        draw_button(screen, fast_track_button, body_font, "Fast Track Train", mouse_pos)
        draw_button(screen, watch_button, body_font, "Watch Every Episode", mouse_pos)
        draw_text(screen, small_font, "Fast Track runs everything automatically and shows the final result.", WHITE, 70, 155)
        pygame.display.flip()
        clock.tick(FPS)


def show_results(final_score, final_reward, episodes):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        screen.fill(BACKGROUND)
        draw_text(screen, title_font, "Training Complete", WHITE, 175, 35)
        draw_text(screen, body_font, f"Episodes: {episodes}", WHITE, 220, 80)
        draw_text(screen, body_font, f"Final score: {final_score}", WHITE, 200, 110)
        draw_text(screen, body_font, f"Final reward: {final_reward}", WHITE, 190, 140)
        draw_text(screen, small_font, "Close the window when you are done reviewing the result.", WHITE, 90, 170)
        pygame.display.flip()
        clock.tick(FPS)


choice = show_menu()

if choice == "fast_track":
    final_score, final_reward = train_agent(agent, game, TOTAL_EPISODES)
    print(f"Training complete. Score: {final_score}, Reward: {final_reward}, Epsilon: {agent.epsilon:.3f}")
    showcase_score, showcase_reward = play_agent_showcase(agent, game, screen, clock)
    print(f"Showcase complete. Score: {showcase_score}, Reward: {showcase_reward}")
    show_results(showcase_score, showcase_reward, TOTAL_EPISODES)
else:
    final_score, final_reward = train_agent(agent, game, TOTAL_EPISODES, screen=screen, clock=clock, render=True)
    print(f"Training complete. Score: {final_score}, Reward: {final_reward}, Epsilon: {agent.epsilon:.3f}")
    show_results(final_score, final_reward, TOTAL_EPISODES)

pygame.quit()