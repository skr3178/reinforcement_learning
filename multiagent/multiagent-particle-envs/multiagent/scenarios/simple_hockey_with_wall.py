import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.adversary = False
                agent.color = np.array([0.25, 0.25, 0.75])
        # add landmarks for goal posts and puck
        goal_posts = [[-0.25, -1.0],
                      [-0.25, 1.0],
                      [0.25, -1.0],
                      [0.25, 1.0]]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if i > 0:
                landmark.collide = True
                landmark.movable = False
                landmark.state.p_pos = np.array(goal_posts[i - 1])
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.collide = True
                landmark.movable = True
        # add landmarks for rink boundary
        # world.landmarks += self.set_boundaries(world)
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i > 0:
                landmark.color = np.array([0.7, 0.7, 0.7])
            else:
                landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
        # set random initial states
        for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # world.landmarks[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        # world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.initial_y = agent.state.p_pos[1]
        world.landmarks[0].state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)  # Start puck within bounds
        world.landmarks[0].state.p_vel = np.random.uniform(-0.5, +0.5, world.dim_p)  # Initial small velocity



    def boundary_reflect(self, world):
        """Handle puck reflection off boundaries"""
        puck = world.landmarks[0]
        border = 1.0  # Boundary limit
        restitution = 1.0  # Energy retained after bounce (0.8 = 80%)

        # Check each dimension
        for i in range(world.dim_p):
            if abs(puck.state.p_pos[i]) > border:
                # Reflect position (keep within bounds)
                puck.state.p_pos[i] = np.sign(puck.state.p_pos[i]) * border

                # Reverse velocity with restitution coefficient
                puck.state.p_vel[i] = -puck.state.p_vel[i] * restitution

                # Small random perturbation to prevent deadlocks
                puck.state.p_vel[i] += np.random.uniform(-0.1, 0.1)

    # return all agents of the blue team
    def blue_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all agents of the red team
    def red_agents(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # First handle puck reflection
        self.boundary_reflect(world)

        # Then calculate normal rewards
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Position checks
        y_pos = agent.state.p_pos[1]

        # Vertical movement constraints: lower and upper as -1.0 and 0.0
        lower_bound = -1.0
        upper_bound = 0.0

        # if y_pos < lower_bound or y_pos > upper_bound:
        #     return -1  # Strong boundary penalty

        # Base reward: puck proximity (scaled up)
        # Equivalent to: (x2-x1)² + (y2-y1)² (no square root)
        rew = -10 * np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))

        # physics based puck return velocities
        puck_vel = world.landmarks[0].state.p_vel
        agent_vel = agent.state.p_vel

        # Velocity alignment term (dot product of agent and puck velocities)
        velocity_alignment = np.dot(agent_vel, puck_vel)

        # Normalize by magnitudes to get directional alignment (range: [-1, 1])
        if np.linalg.norm(agent_vel) > 0.1 and np.linalg.norm(puck_vel) > 0.1:
            alignment_score = velocity_alignment / (
                    np.linalg.norm(agent_vel) * np.linalg.norm(puck_vel)
            )
        else:
            alignment_score = 0  # Ignore if velocities are too small
        rew += alignment_score

        # Goal detection
        puck_x, puck_y = world.landmarks[0].state.p_pos
        if -0.25 <= puck_x <= 0.25:  # Within goal width
            if (puck_y >= 1.0):
                rew += 30  # Scored goal
            elif (puck_y <= -1.0):
                rew -= 30  # Conceded goal

        # making sure agents/puck do not escape from the bounding box
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     # return min(np.exp(2 * x - 2), 10)
        #     return 10
        # # Apply penalty to each dimension
        # bounding_penalty = 0
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     bounding_penalty += bound(x)


        return rew

    def adversary_reward(self, agent, world):
        y_pos = agent.state.p_pos[1]
        initial_y = agent.initial_y

        # Vertical movement constraints: lower and upper as 1.0 and 0.0
        lower_bound = 1.0
        upper_bound = 0.0

        # if y_pos < lower_bound or y_pos > upper_bound:
        #     return -1  # Strong boundary penalty

        # Base reward: puck proximity (scaled up)
        rew_2 = -10 * np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))

        # physics based puck return velocities
        puck_vel = world.landmarks[0].state.p_vel
        agent_vel = agent.state.p_vel

        # Velocity alignment term (dot product of agent and puck velocities)
        velocity_alignment = np.dot(agent_vel, puck_vel)

        # Normalize by magnitudes to get directional alignment (range: [-1, 1])
        if np.linalg.norm(agent_vel) > 0.1 and np.linalg.norm(puck_vel) > 0.1:
            alignment_score = velocity_alignment / (
                    np.linalg.norm(agent_vel) * np.linalg.norm(puck_vel)
            )
        else:
            alignment_score = 0  # Ignore if velocities are too small
        rew_2 += alignment_score

        # Goal detection
        puck_x, puck_y = world.landmarks[0].state.p_pos
        if -0.25 <= puck_x <= 0.25:  # Within goal width
            if (puck_y <= -1.0):
                rew_2 += 30  # Scored goal
            elif (puck_y >= 1.0):
                rew_2 -= 30  # Conceded goal

        return rew_2 # True zero-sum

    def observation(self, agent, world):
        # get positions/vel of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            if entity.movable:
                entity_vel.append(entity.state.p_vel)
        # get positions/vel of all other agents in this agent's reference frame
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + entity_pos + entity_vel + other_pos + other_vel)