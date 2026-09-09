import math
import itertools
from rlbot.flat import *
from rlbot.managers import Bot

class Matrix3:
    __slots__ = ('data', 'forward', 'right', 'up')
    def __init__(self, pitch=0, yaw=0, roll=0):
        CP = math.cos(pitch)
        SP = math.sin(pitch)
        CY = math.cos(yaw)
        SY = math.sin(yaw)
        CR = math.cos(roll)
        SR = math.sin(roll)
        self.data = (
            Vector(CP * CY, CP * SY, SP),
            Vector(CY * SP * SR - CR * SY, SY * SP * SR + CR * CY, -CP * SR),
            Vector(-CR * CY * SP - SR * SY, -CR * SY * SP + SR * CY, CP * CR)
        )
        self.forward, self.right, self.up = self.data

    def dot(self, vector):
        return Vector(self.forward.dot(vector), self.right.dot(vector), self.up.dot(vector))

class Vector:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __eq__(self, value):
        return abs(self.x - value.x) < 1e-9 and abs(self.y - value.y) < 1e-9 and abs(self.z - value.z) < 1e-9

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, value):
        return Vector(self.x + value.x, self.y + value.y, self.z + value.z)
    
    __radd__ = __add__

    def __sub__(self, value):
        return Vector(self.x - value.x, self.y - value.y, self.z - value.z)

    def __mul__(self, value):
        return Vector(self.x * value, self.y * value, self.z * value)
    
    __rmul__ = __mul__

    def __truediv__(self, value):
        return Vector(self.x / value, self.y / value, self.z / value)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, value: 'Vector') -> float:
        return self.x * value.x + self.y * value.y + self.z * value.z

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag != 0:
            return Vector(self.x / mag, self.y / mag, self.z / mag)
        return Vector()

    def flatten(self) -> 'Vector':
        return Vector(self.x, self.y, 0.0)

    def dist(self, value: 'Vector') -> float:
        return math.sqrt((self.x - value.x)**2 + (self.y - value.y)**2 + (self.z - value.z)**2)

    def dist_sq(self, value: 'Vector') -> float:
        return (self.x - value.x)**2 + (self.y - value.y)**2 + (self.z - value.z)**2

    def angle(self, other: 'Vector') -> float:
        denom = self.magnitude() * other.magnitude()
        if denom == 0:
            return 0.0
        dotv = max(-1.0, min(1.0, self.dot(other) / denom))
        return math.acos(dotv)
    
    def __repr__(self):
        return f"Vector(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class CarObject:
    __slots__ = ('location', 'orientation', 'velocity', 'angular_velocity', 'demolished', 'airborne', 'jumped', 'doublejumped', 'boost', 'index', 'team')
    def __init__(self, index):
        self.location = Vector()
        self.orientation = Matrix3()
        self.velocity = Vector()
        self.angular_velocity = Vector()
        self.demolished = False
        self.airborne = False
        self.jumped = False
        self.doublejumped = False
        self.boost = 0
        self.index = index
        self.team = 0

    def update(self, packet: GamePacket, car_index=None):
        if car_index is None:
            car_index = self.index
        if car_index >= len(packet.players) or car_index < 0:
            return
            
        car = packet.players[car_index]
        car_phy = car.physics
        self.location = Vector(car_phy.location.x, car_phy.location.y, car_phy.location.z)
        self.velocity = Vector(car_phy.velocity.x, car_phy.velocity.y, car_phy.velocity.z)
        self.orientation = Matrix3(car_phy.rotation.pitch, car_phy.rotation.yaw, car_phy.rotation.roll)
        self.angular_velocity = self.orientation.dot(Vector(car_phy.angular_velocity.x, car_phy.angular_velocity.y, car_phy.angular_velocity.z))
        
        self.demolished = car.demolished_timeout > 0
        self.airborne = car.air_state != AirState.OnGround
        self.jumped = car.has_jumped
        self.doublejumped = car.has_double_jumped
        self.boost = car.boost
        self.index = car_index
        self.team = car.team

    def local(self, value): 
        return self.orientation.dot(value)   
    @property
    def forward(self): return self.orientation.data[0]
    # data[1] from the pitch/yaw/roll rotation matrix is the car's RIGHT
    # vector (standard convention for this matrix formula); `left` is its
    # negation.
    @property
    def right(self): return self.orientation.data[1]
    @property
    def left(self): return self.orientation.data[1] * -1
    @property
    def up(self): return self.orientation.data[2]


class BallObject:
    __slots__ = ('location', 'velocity')
    def __init__(self):
        self.location = Vector()
        self.velocity = Vector()
    def update(self, packet: GamePacket):
        if len(packet.balls) == 0:
            return
        ball = packet.balls[0]
        self.location = Vector(ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)
        self.velocity = Vector(ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z)


def clamp(n, _min, _max):
    return max(_min, min(_max, n))

def adaptive_slice_indices(max_index, near_limit=240, near_step=2, far_step=5):
    """Indices into a ball prediction, spanning [0, max_index) instead of an
    arbitrary truncated window. Sampling is dense (near_step) for the near
    future, where precision matters most and where a match is most likely to
    be found, and sparse (far_step) further out, where a coarser look is
    enough since the loop that consumes these usually exits (via early-break)
    the moment it finds a usable slice anyway. This gives noticeably better
    coverage of the full ~5s prediction for roughly the same, or fewer, total
    samples as a flat step would need to reach that far.
    """
    near_limit = min(near_limit, max_index)
    return itertools.chain(range(0, near_limit, near_step), range(near_limit, max_index, far_step))
    
def steerPD(angle, rate):
    STEER_P = 5.0
    STEER_D = 0.3
    output = (angle * STEER_P) + (rate * STEER_D)
    
    return clamp(output, -1, 1)

def defaultPD(agent, local_target, upside_down=False, up=None):
    if up is None:
        up = agent.me.local(Vector(z=-1 if upside_down else 1))
    target_angles = (
        math.atan2(local_target.z, local_target.x),
        math.atan2(local_target.y, local_target.x),
        math.atan2(up.y, up.z)
    )
    agent.controller_state.pitch = steerPD(target_angles[0], agent.me.angular_velocity.y / 4)
    agent.controller_state.yaw = steerPD(target_angles[1], -agent.me.angular_velocity.z / 4)
    agent.controller_state.roll = steerPD(target_angles[2], agent.me.angular_velocity.x / 4)
    return target_angles


# =============================================================================
# ADVANCED AERIAL CONTROLLER - OPTIMAL CONTROL BASED
# =============================================================================
# Combines:
# 1. CasADI-inspired direct collocation for optimal trajectory planning
# 2. NTUD-style 1D kinematic controller for minimum-time interception
# 3. Nose-targeting for precise ball contact
# =============================================================================

GRAVITY = 650.0
BOOST_ACCEL = 991.666
THROTTLE_ACCEL = 66.666
JUMP_SPEED = 291.666
JUMP_ACCEL = 1458.333
JUMP_MAX_DURATION = 0.2
MASS = 180.0

# Car physics constants
MAX_CAR_SPEED = 2300.0
AERIAL_DRAG = 0.03

def solve_1d_minimum_time(pos0, vel0, target_pos, target_vel, max_accel, gravity=0.0):
    """
    NTUD-style 1D kinematic minimum-time solver.
    Solves: min T s.t. pos(T) = target_pos, vel(T) = target_vel
    with |accel| <= max_accel, using bang-bang control.
    Returns (time, accel_profile) or (inf, None) if unreachable.
    """
    # Simplified: assume we can reach target_vel = 0 at intercept
    # For aerial, we solve the double integrator with gravity compensation
    
    delta_pos = target_pos - pos0
    delta_vel = target_vel - vel0
    
    # With gravity compensation, effective acceleration is max_accel - gravity
    # (gravity acts against upward motion)
    a_max = max_accel
    a_min = -max_accel  # can always thrust downward
    
    # Time-optimal 1D trajectory with bounded acceleration (bang-bang)
    # We need to find switching times for acceleration profile
    
    # For minimum time to reach position with zero final velocity:
    # Use the classical time-optimal control solution
    # v^2 = v0^2 + 2*a*(x - x0)
    
    # If we accelerate at a_max then decelerate at a_min:
    # Distance during accel: d1 = (v_peak^2 - v0^2) / (2*a_max)
    # Distance during decel: d2 = (0 - v_peak^2) / (2*a_min) = v_peak^2 / (2*|a_min|)
    # Total: d = d1 + d2 = (v_peak^2 - v0^2)/(2*a_max) + v_peak^2/(2*|a_min|)
    # If a_max = |a_min| = a: d = v_peak^2/a - v0^2/(2a)
    # So v_peak^2 = a*d + v0^2/2
    # Time = (v_peak - v0)/a + v_peak/a = (2*v_peak - v0)/a
    
    a = a_max
    required_disp = delta_pos
    
    # Check if we can do it with pure acceleration (no coast)
    # v_peak^2 = v0^2 + 2*a*d1, and v_peak^2 = 2*a*d2 (decel to 0)
    # So d1 = (v_peak^2 - v0^2)/(2a), d2 = v_peak^2/(2a)
    # d = d1 + d2 = v_peak^2/a - v0^2/(2a)
    # v_peak^2 = a*d + v0^2/2
    
    v_peak_sq = a * abs(required_disp) + vel0**2 / 2
    if v_peak_sq < 0:
        return float('inf'), None
    
    v_peak = math.sqrt(v_peak_sq)
    t_accel = (v_peak - vel0) / a if required_disp >= 0 else (v_peak + vel0) / a
    t_decel = v_peak / a
    
    if t_accel < 0:
        # Need to decelerate first
        t_accel = 0
        v_peak = abs(vel0)
        t_decel = v_peak / a
        # Check if we can reach target
        d_decel = vel0**2 / (2*a)
        if abs(required_disp) > d_decel:
            return float('inf'), None
    
    return t_accel + t_decel, (t_accel, t_decel, v_peak)


def direct_collocation_trajectory(car_pos, car_vel, car_up, ball_pos, ball_vel, intercept_time, dt=1/120.0):
    """
    CasADI-inspired direct collocation for optimal aerial trajectory.
    Discretizes time into N steps, optimizes control inputs (thrust direction, boost)
    to minimize: ||pos_N - ball_pos||^2 + w * ||vel_N||^2
    subject to: car dynamics, control bounds.
    
    This is a simplified implementation using iterative LQR / shooting method
    since we can't use CasADI directly in RLBot.
    """
    N = int(intercept_time / dt)
    if N < 2:
        N = 2
    if N > 120:
        N = 120
        dt = intercept_time / N
    
    gravity_vec = Vector(0, 0, -GRAVITY)
    
    # Initialize with simple ballistic + thrust toward target
    states = []
    controls = []
    
    pos = Vector(car_pos.x, car_pos.y, car_pos.z)
    vel = Vector(car_vel.x, car_vel.y, car_vel.z)
    
    # Forward pass: generate nominal trajectory
    for i in range(N):
        t = i * dt
        # Predict ball position at this time
        ball_pred = ball_pos + ball_vel * t
        
        # Direction to ball
        to_ball = (ball_pred - pos).normalize()
        
        # Control: thrust toward ball with boost when aligned
        thrust_dir = to_ball
        use_boost = vel.dot(thrust_dir) > 500
        
        accel = thrust_dir * (BOOST_ACCEL if use_boost else THROTTLE_ACCEL)
        accel += gravity_vec
        
        states.append((pos, vel))
        controls.append((thrust_dir, use_boost))
        
        vel = vel + accel * dt
        pos = pos + vel * dt
    
    # Backward pass: refine using gradient descent on final error
    final_error = pos - (ball_pos + ball_vel * intercept_time)
    
    # Simple correction: adjust initial thrust direction
    # This is a simplified shooting method
    for iteration in range(3):
        # Re-simulate with corrected initial aim
        correction = -final_error * 0.3 / N
        
        pos = Vector(car_pos.x, car_pos.y, car_pos.z)
        vel = Vector(car_vel.x, car_vel.y, car_vel.z)
        
        for i in range(N):
            t = i * dt
            ball_pred = ball_pos + ball_vel * t
            to_ball = (ball_pred - pos).normalize()
            
            # Apply correction mostly early in trajectory
            if i < N // 3:
                thrust_dir = (to_ball + correction).normalize()
            else:
                thrust_dir = to_ball
            
            use_boost = vel.dot(thrust_dir) > 500 and i < N * 0.7
            accel = thrust_dir * (BOOST_ACCEL if use_boost else THROTTLE_ACCEL) + gravity_vec
            
            vel = vel + accel * dt
            pos = pos + vel * dt
        
        final_error = pos - (ball_pos + ball_vel * intercept_time)
        if final_error.magnitude() < 50:
            break
    
    # Return the first control action
    if controls:
        return controls[0][0], controls[0][1], states
    return Vector(0, 0, 1), False, []


class Maneuver_Aerial:
    """
    Advanced aerial controller using optimal control theory.
    
    Key improvements over previous version:
    - NTUD 1D kinematic solver for minimum-time intercept feasibility
    - CasADI-inspired direct collocation for optimal trajectory
    - Nose-targeting: aligns car's forward vector with required acceleration
    - Adaptive intercept time selection using reachability analysis
    - Explicit boost management for time-optimal trajectories
    """
    def __init__(self, agent, intercept_time, shot_target):
        self.intercept_time = intercept_time
        self.shot_target = shot_target
        self.ball_location = None
        self.ball_velocity = None
        
        # Physics constants
        self.jump_speed = JUMP_SPEED
        self.jump_acc = JUMP_ACCEL
        self.jump_max_duration = JUMP_MAX_DURATION
        self.boost_accel = BOOST_ACCEL
        self.throttle_accel = THROTTLE_ACCEL
        
        # State
        self.jumping = True
        self.jump_time = agent.time
        self._jump_impulse_applied = False
        self._trajectory_plan = None
        self._plan_valid_until = 0.0
        self._last_intercept_time = intercept_time
        
        # Nose targeting
        self._target_nose_dir = None
        self._align_phase = False

    def _compute_optimal_intercept(self, agent, prediction, current_T):
        """
        Find the optimal intercept point using reachability analysis.
        Uses NTUD-style 1D kinematic checks in each axis.
        """
        current_slice_index = math.ceil(current_T * 120)
        NARROW_WINDOW = 60
        WIDE_WINDOW = 180
        
        gravity = Vector(0, 0, -GRAVITY)
        
        def check_reachability(slice_idx, dt):
            """Check if car can reach ball at this slice using 1D kinematics."""
            s = prediction.slices[slice_idx]
            ball_pos = Vector(s.physics.location.x, s.physics.location.y, s.physics.location.z)
            ball_vel = Vector(s.physics.velocity.x, s.physics.velocity.y, s.physics.velocity.z)
            
            # Car predicted position without control
            car_pos = agent.me.location + agent.me.velocity * dt + gravity * 0.5 * dt * dt
            car_vel = agent.me.velocity + gravity * dt
            
            # Check each axis independently (NTUD approach)
            # X-axis
            tx, _ = solve_1d_minimum_time(car_pos.x, car_vel.x, ball_pos.x, ball_vel.x, self.boost_accel)
            # Y-axis  
            ty, _ = solve_1d_minimum_time(car_pos.y, car_vel.y, ball_pos.y, ball_vel.y, self.boost_accel)
            # Z-axis (with gravity)
            tz, _ = solve_1d_minimum_time(car_pos.z, car_vel.z, ball_pos.z, ball_vel.z, self.boost_accel, GRAVITY)
            
            # Overall time is max of axes (coupled through thrust direction)
            t_min = max(tx, ty, tz)
            
            # Account for turning time
            to_ball = (ball_pos - car_pos).normalize()
            turn_angle = agent.me.forward.angle(to_ball)
            turn_time = turn_angle * 0.35  # empirical
            
            return t_min + turn_time <= dt + 0.1, ball_pos, ball_vel, t_min
        
        def find_best(window):
            start = clamp(current_slice_index - window, 0, len(prediction.slices) - 1)
            end = clamp(current_slice_index + window, 0, len(prediction.slices) - 1)
            best = None
            best_score = float('inf')
            
            for i in range(start, end + 1):
                s = prediction.slices[i]
                dt = s.game_seconds - agent.time
                if dt <= 0.05:
                    continue
                
                reachable, ball_pos, ball_vel, t_min = check_reachability(i, dt)
                if not reachable:
                    continue
                
                # Score: prefer earlier intercepts, penalize high altitude
                ball_pos = Vector(s.physics.location.x, s.physics.location.y, s.physics.location.z)
                altitude_penalty = max(0, (ball_pos.z - 500) / 1000.0)
                score = dt + altitude_penalty * 0.5
                
                if score < best_score:
                    best_score = score
                    best = (s.game_seconds, ball_pos, ball_vel, t_min)
            
            return best
        
        result = find_best(NARROW_WINDOW)
        if result is None:
            result = find_best(WIDE_WINDOW)
        
        return result

    def _plan_trajectory(self, agent, packet):
        """
        Plan optimal trajectory using direct collocation (CasADI-inspired).
        Called periodically or when intercept target changes significantly.
        """
        if self.ball_location is None:
            return None
        
        T = self.intercept_time - agent.time
        if T <= 0.1:
            return None
        
        # Check if we need to replan
        if self._trajectory_plan is not None and agent.time < self._plan_valid_until:
            if abs(self.intercept_time - self._last_intercept_time) < 0.1:
                return self._trajectory_plan
        
        # Plan using direct collocation
        thrust_dir, use_boost, states = direct_collocation_trajectory(
            agent.me.location, agent.me.velocity, agent.me.up,
            self.ball_location, self.ball_velocity, T
        )
        
        self._trajectory_plan = (thrust_dir, use_boost, states)
        self._plan_valid_until = agent.time + 0.1
        self._last_intercept_time = self.intercept_time
        
        # Compute nose targeting direction
        # The nose should point along the required acceleration vector
        gravity = Vector(0, 0, -GRAVITY)
        required_accel = thrust_dir * (BOOST_ACCEL if use_boost else THROTTLE_ACCEL)
        total_accel = required_accel + gravity
        
        # Nose should align with total acceleration for optimal control
        if total_accel.magnitude() > 10:
            self._target_nose_dir = total_accel.normalize()
        else:
            self._target_nose_dir = thrust_dir
        
        return self._trajectory_plan

    def __call__(self, agent, packet):
        prediction = agent.get_ball_prediction()
        if prediction is None or len(prediction.slices) == 0:
            agent.maneuver_lock = 0
            return
        
        current_T = self.intercept_time - agent.time
        
        # Re-evaluate intercept point every frame using optimal control
        intercept_result = self._compute_optimal_intercept(agent, prediction, current_T)
        
        if intercept_result is not None:
            best_time, best_ball_loc, best_ball_vel, t_min = intercept_result
            self.ball_location = best_ball_loc
            self.ball_velocity = best_ball_vel
            self.intercept_time = best_time
            current_T = best_time - agent.time
        
        # Abort conditions
        if self.ball_location is None:
            agent.maneuver_lock = 0
            return
        
        if current_T < 0.05:
            agent.maneuver_lock = 0
            return
        
        # Plan trajectory
        plan = self._plan_trajectory(agent, packet)
        if plan is None:
            agent.maneuver_lock = 0
            return
        
        thrust_dir, use_boost, states = plan
        
        T = self.intercept_time - agent.time
        gravity = Vector(0, 0, -GRAVITY)
        
        # Jump phase management
        jump_elapsed = agent.time - self.jump_time
        
        if self.jumping:
            tau = self.jump_max_duration - jump_elapsed
            
            if not self._jump_impulse_applied:
                self._jump_impulse_applied = True
            
            if jump_elapsed <= self.jump_max_duration:
                agent.controller_state.jump = True
            else:
                self.jumping = False
                self._align_phase = True  # Start nose alignment after jump
        
        # Target for rendering
        shot_vector = (self.shot_target - self.ball_location).normalize()
        ball_radius_offset = 90.0
        target = self.ball_location - shot_vector * ball_radius_offset
        agent.renderer_target = target
        
        # Get car's predicted state at intercept (ballistic)
        xf = agent.me.location + agent.me.velocity * T + gravity * 0.5 * T * T
        
        # Vector from predicted car pos to target
        delta_x = target - xf
        
        # NOSE TARGETING: Instead of aiming at target, aim the nose along required acceleration
        if self._target_nose_dir is not None and self._align_phase:
            # Aim point is car position + nose direction (makes nose point along accel)
            aim_point = agent.me.location + self._target_nose_dir * 1000
            up_ref = Vector(0, 0, 1)
            cubic_aim_at(agent, aim_point, up=up_ref)
        else:
            # Fallback to cubic aim during jump phase
            aim_point = agent.me.location + delta_x
            up_ref = Vector(0, 0, 1)
            cubic_aim_at(agent, aim_point, up=up_ref)
        
        # Boost / throttle from optimal plan
        required_speed = delta_x.dot(agent.me.forward) / T if T > 0 else 0
        
        if use_boost and agent.me.boost > 0 and required_speed >= (self.boost_accel + self.throttle_accel) / 30.0:
            agent.controller_state.boost = True
            required_speed -= self.boost_accel / 30.0
        else:
            agent.controller_state.boost = False
        
        if T > 0:
            agent.controller_state.throttle = clamp(required_speed / (self.throttle_accel / 30.0), -1, 1)
        
        # Final approach: flip into ball with nose
        if not agent.me.doublejumped and T < 0.15:
            # Compute desired hit direction (from ball to shot target)
            hit_dir = (self.shot_target - self.ball_location).normalize()
            
            # Nose should point along hit_dir at impact
            local_hit = agent.me.local(hit_dir).normalize()
            
            pitch_angle = math.atan2(local_hit.z, local_hit.x)
            yaw_angle = math.atan2(local_hit.y, local_hit.x)
            
            # Strong correction to align nose
            agent.controller_state.pitch = clamp(pitch_angle * 5.0, -1.0, 1.0)
            agent.controller_state.yaw = clamp(yaw_angle * 5.0, -1.0, 1.0)
            agent.controller_state.jump = True
            
            # Roll to keep wheels down
            local_up = agent.me.local(Vector(0, 0, 1))
            roll_angle = math.atan2(local_up.y, local_up.z)
            agent.controller_state.roll = clamp(roll_angle * 3.0, -1.0, 1.0)
        
        if T <= -0.4:
            agent.maneuver_lock = 0.0

FIELD_LENGTH = 10240.0
FIELD_WIDTH = 8192.0
GOAL_WIDTH = 1786.0
GOAL_HEIGHT = 642.0

# Kickoff spawn classification threshold (|X| < this → central spawn → central kickoff)
KICKOFF_CENTRAL_X = 600
AERIAL_START_ANGLE = math.pi / 4

def pos(obj) -> Vector:
    if hasattr(obj, 'physics'):
        return Vector(obj.physics.location.x, obj.physics.location.y, obj.physics.location.z)
    elif hasattr(obj, 'location'):
        return Vector(obj.location.x, obj.location.y, obj.location.z)
    return None

def constrain_pi(n):
    return math.remainder(n, 2 * math.pi)


def local_coordinates(car: CarObject, target: Vector) -> Vector:
    return car.local(target - car.location)

def drive_backwards_to_target(agent, local_target: Vector, target_speed: float, up: Vector):
    target_angles = defaultPD(agent, local_target, up=up)
    yaw_angle = target_angles[1]

    if agent.me.airborne:
        agent.controller_state.steer = 0
    else:
        agent.controller_state.steer = -steerPD(yaw_angle, 0)

    reverse_speed = -agent.me.velocity.dot(agent.me.forward)
    if reverse_speed < target_speed:
        agent.controller_state.throttle = -1.0
    elif reverse_speed - target_speed > 100 or (target_speed < 10 and reverse_speed > 10):
        agent.controller_state.throttle = 1.0
    else:
        agent.controller_state.throttle = 0.0

    agent.controller_state.boost = False
    agent.controller_state.handbrake = False
    agent.drive_backwards = True


def should_reverse_to_target(agent, local_target: Vector, target_pos: Vector, target_speed: float, is_target_on_wall: bool, reverse_distance: float = 2500.0) -> bool:
    if agent.me.airborne or agent.me.jumped or agent.me.doublejumped:
        return False
    if is_target_on_wall:
        return False

    surface_normal = agent.get_surface_normal(agent.me.location)
    if surface_normal.z < 0.99:
        return False

    local_flat = Vector(local_target.x, local_target.y, 0)
    dist_to_target = local_flat.magnitude()
    if dist_to_target > reverse_distance:
        return False
    if dist_to_target < 10:
        return False

    yaw_angle = math.atan2(local_flat.y, local_flat.x)
    if local_flat.x >= -120 or abs(yaw_angle) < 1.0:
        return False

    current_reverse_speed = max(0.0, -agent.me.velocity.dot(agent.me.forward))
    reverse_eta = dist_to_target / max(1000.0, current_reverse_speed + 1000.0)
    turn_eta = abs(constrain_pi(yaw_angle)) * 0.35 + dist_to_target / 1700.0

    if target_speed < 500 and dist_to_target > 700:
        return False

    return reverse_eta < turn_eta


def drive_to_target(agent, target_pos: Vector, target_speed: float, boost: bool = False, reverse_distance: float = 2500.0, ramp_down: bool = False):
    agent.renderer_target = target_pos
    local_target = local_coordinates(agent.me, target_pos)

    surface_normal = agent.get_surface_normal(agent.me.location)
    up = agent.me.local(surface_normal)

    # Enhanced wall play: if target is on wall, don't ignore vertical distance for better wall approach
    is_target_on_wall = agent.is_ball_on_wall(target_pos)
    if not agent.me.airborne and not is_target_on_wall:
        local_target.z = 0 # Flatten target for ground steering

    agent.drive_backwards = False
    if should_reverse_to_target(agent, local_target, target_pos, target_speed, is_target_on_wall, reverse_distance):
        drive_backwards_to_target(agent, local_target, target_speed, up)
        return

    target_angles = defaultPD(agent, local_target, up=up)

    yaw_angle = target_angles[1]
    
    if agent.me.airborne:
        agent.controller_state.steer = 0
    else:
        agent.controller_state.steer = steerPD(yaw_angle, 0)
 
    # FIX: Calculate speed relative to the car's nose instead of absolute magnitude
    current_speed = agent.me.velocity.dot(agent.me.forward)
    
    # --- Distance-based speed ramp-down for point targets (e.g. boost pads) ---
    # When approaching a point target the car must decelerate *before*
    # reaching it. Without this the car blows past the pad, overrotates,
    # and spirals in circles — repeatedly overshooting the pickup without
    # ever slowing down enough to collect the boost. We cap the target speed
    # to the maximum velocity from which we can still stop within the
    # remaining horizontal distance (~1400 uu/s² full-brake deceleration).
    original_target_speed = target_speed
    if ramp_down and not agent.me.airborne:
        dist_to_target = local_target.magnitude()
        if dist_to_target < 2000.0:
            decel_rate = 1400.0
            max_safe_speed = math.sqrt(2.0 * decel_rate * max(dist_to_target, 1.0))
            target_speed = min(target_speed, max(max_safe_speed, 300.0))
    
    if current_speed < target_speed:
        agent.controller_state.throttle = 1.0
        
        # Don't boost when deliberately targeting low speeds (<1200),
        # e.g. slowing down to get underneath a falling ball.
        # Cone tightened from 0.7 -> 0.5 rad: boost accelerates in the
        # direction the car is *facing*, not the direction it's aimed at the
        # target, so boosting while turned much more than ~29 degrees off
        # target wastes boost pushing the car off its intended line instead
        # of toward it.
        if (not agent.me.airborne and abs(yaw_angle) < 0.5 and boost
                and current_speed < 2250 and target_speed >= 1200):
             agent.controller_state.boost = True
        else:
             agent.controller_state.boost = False
    else:
        speed_diff = current_speed - target_speed
        # Brake if speed overshoot is significant, or we need to stop but are still moving.
        if speed_diff > 100 or (target_speed < 10 and current_speed > 10):
            agent.controller_state.throttle = -1.0 
        else:
            agent.controller_state.throttle = 0.0
        agent.controller_state.boost = False

    handbrake_threshold = math.pi * 0.5 if is_target_on_wall else math.pi * 0.6
    forward_speed = max(0.0, current_speed)
    brake_turn = (
        not agent.me.airborne
        and forward_speed > 850
        and abs(yaw_angle) > 0.75
        and target_speed >= 900
    )
    agent.controller_state.handbrake = brake_turn or (not agent.me.airborne and abs(yaw_angle) > handbrake_threshold)
    if brake_turn:
        agent.controller_state.throttle = -1.0
        agent.controller_state.steer = 1.0 if yaw_angle > 0 else -1.0
    if agent.controller_state.handbrake:
        agent.controller_state.boost = False

    # Trigger a speed-up flip when driving straight toward a distant target.
    agent.try_front_flip_to_target(target_pos, yaw_angle)

def turn_radius(speed):
    speed = max(0.01, min(speed, 2300.0))
    if speed <= 500:
        return (speed / 500) * (251 - 145) + 145
    elif speed <= 1000:
        return ((speed - 500) / 500) * (425 - 251) + 251
    elif speed <= 1500:
        return ((speed - 1000) / 500) * (727 - 425) + 425
    elif speed <= 1750:
        return ((speed - 1500) / 250) * (909 - 727) + 727
    else:
        return ((speed - 1750) / 550) * (1136 - 909) + 909


def vec_flat_direction(v1, v2):
    return (v2 - v1).flatten().normalize()


def vec3_rotate(v, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return Vector(
        v.x * c + v.y * s,
        v.y * c - v.x * s,
        v.z
    )


def cubic_steerPD(angle, rate):
    val = (35.0 * (angle + rate)) ** 3 / 10.0
    return max(-1.0, min(1.0, val))


def cubic_aim_at(agent, target_location, up=None, backwards=False):
    local_target = agent.me.local(target_location - agent.me.location)
    if backwards:
        local_target = -local_target
    
    if up is None:
        up = Vector(0, 0, 1)
    
    local_up = agent.me.local(up.normalize())
    
    target_angles = (
        math.atan2(local_target.z, local_target.x),
        math.atan2(local_target.y, local_target.x),
        math.atan2(local_up.y, local_up.z)
    )
    
    agent.controller_state.steer = cubic_steerPD(target_angles[1], -agent.me.angular_velocity.z * 0.01) * (-1 if backwards else 1)
    agent.controller_state.pitch = cubic_steerPD(target_angles[0], agent.me.angular_velocity.y * 0.2)
    agent.controller_state.yaw = cubic_steerPD(target_angles[1], -agent.me.angular_velocity.z * 0.15)
    agent.controller_state.roll = cubic_steerPD(target_angles[2], agent.me.angular_velocity.x * 0.25)
    return target_angles


def in_field(pos, radius):
    point_x = abs(pos.x)
    point_y = abs(pos.y)
    point_z = abs(pos.z)
    if point_x > FIELD_WIDTH / 2.0 - radius:
        return False
    if point_y > FIELD_LENGTH / 2.0 + 380.0 - radius:
        return False
    if (point_x > GOAL_WIDTH / 2.0 - radius or point_z > GOAL_HEIGHT - radius) and point_y > FIELD_LENGTH / 2.0 - radius:
        return False
    if point_x + point_y > 8064.0 - radius:
        return False
    return True


class SpeedFlip:
    def __init__(self, direction):
        self.finished = False
        self.direction = direction.normalize()
        self.start_time = -1.0
        self.side = 0

    def run(self, agent):
        agent.controller_state.throttle = 1.0

        if self.start_time < 0:
            forward_vel = agent.me.velocity.dot(agent.me.forward)
            tr = turn_radius(forward_vel)
            angle = 0.06 * forward_vel / tr if tr > 0.01 else 0.0
            
            left_vec = vec3_rotate(self.direction, angle).flatten().normalize()
            right_vec = vec3_rotate(self.direction, -angle).flatten().normalize()

            angle_left = agent.me.velocity.angle(left_vec)
            angle_right = agent.me.velocity.angle(right_vec)

            if angle_left < angle_right:
                cubic_aim_at(agent, agent.me.location + left_vec)

                flat_angle_left = agent.me.velocity.flatten().angle(left_vec)
                if flat_angle_left < 0.05:
                    self.start_time = agent.time
                    self.side = 1  # Right
            else:
                cubic_aim_at(agent, agent.me.location + right_vec)

                flat_angle_right = agent.me.velocity.flatten().angle(right_vec)
                if flat_angle_right < 0.05:
                    self.start_time = agent.time
                    self.side = -1  # Left
        else:
            elapsed = agent.time - self.start_time

            if 0 < elapsed < 0.1:
                agent.controller_state.jump = True
            elif 0.12 < elapsed < 0.15:
                agent.controller_state.jump = True
                agent.controller_state.pitch = -1.0
                agent.controller_state.roll = self.side * 0.5
            elif 0.15 < elapsed < 0.75:
                agent.controller_state.pitch = 1.0
                agent.controller_state.roll = float(self.side)
            elif 0.75 < elapsed < 0.9:
                agent.controller_state.pitch = 1.0
                agent.controller_state.handbrake = True
                agent.controller_state.roll = float(self.side)
                agent.controller_state.yaw = float(self.side)
            elif 0.9 < elapsed:
                self.finished = True

        if not in_field(agent.me.location, 150) or (self.start_time < 0 and agent.me.airborne):
            self.finished = True


def Maneuver_FrontDodge(agent, packet):
    if getattr(agent, "_front_dodge_start_time", -1.0) < 0.0:
        agent._front_dodge_start_time = agent.time
        agent._front_dodge_jumping = not agent.me.airborne

    direction = getattr(agent, "_front_dodge_direction", Vector(0, -1 if agent.team == 0 else 1, 0)).normalize()
    jump_time = getattr(agent, "_front_dodge_jump_time", 0.18)
    elapsed = agent.time - agent._front_dodge_start_time
    jumping = getattr(agent, "_front_dodge_jumping", not agent.me.airborne)

    agent.controller_state.throttle = 1.0
    agent.controller_state.boost = False

    if agent.me.airborne and elapsed > (jump_time if jumping else 0.0) + 0.1:
        agent.maneuver_lock = 0.0
        return

    if elapsed < jump_time and jumping:
        agent.controller_state.jump = True
    elif getattr(agent, "_front_dodge_step", 0) < 3 and jumping:
        agent.controller_state.jump = False
        agent._front_dodge_step = getattr(agent, "_front_dodge_step", 0) + 1
    elif elapsed < (jump_time if jumping else 0.0) + 0.6:
        if getattr(agent, "_front_dodge_input", None) is None:
            forward = agent.me.forward.flatten().normalize()
            right = agent.me.right.flatten().normalize()
            local_x = right.dot(direction)
            local_y = -forward.dot(direction)
            forward_vel = agent.me.velocity.dot(agent.me.forward)
            speed_ratio = abs(forward_vel) / 2300.0
            backwards_dodge = (
                local_x < 0.0
                if abs(forward_vel) < 100.0
                else (local_x >= 0.0) != (forward_vel > 0.0)
            )
            if backwards_dodge:
                local_x /= (16.0 / 15.0) * (1.0 + 1.5 * speed_ratio)
            local_y /= 1.0 + 0.9 * speed_ratio
            agent._front_dodge_input = Vector(local_x, local_y, 0).normalize()
        agent.controller_state.yaw = clamp(agent._front_dodge_input.x, -1.0, 1.0)
        agent.controller_state.pitch = clamp(agent._front_dodge_input.y, -1.0, 1.0)
        agent.controller_state.jump = True
    else:
        agent.maneuver_lock = 0.0


class Kickoff:
    def __init__(self, agent, packet, kickoff_type='speed'):
        self.flip_start_time = 0.0
        self.type = kickoff_type
        self._is_diagonal = False
        self._time_on_ground = 0.0
        self._speed_flipped = False
        self._speed_flip = None
        self.state = 'approach'
        label = 'DodgeFlip' if self.type == 'central' else 'Speedflip'
        try:
            agent.send_quickchat(self.type, f"{label} kickoff!")
        except Exception:
            pass

    def get_output(self, agent, packet):
        if len(packet.balls) == 0:
            return
        ball_pos = pos(packet.balls[0])

        if agent.me.velocity.magnitude() < 200:
            self._is_diagonal = abs(agent.me.location.x) > 1000

        if self._speed_flipped and not agent.me.airborne:
            self._time_on_ground += agent.delta

        if self._speed_flip is not None and not self._speed_flip.finished:
            agent.controller_state.boost = True
            self._speed_flip.run(agent)
        else:
            agent.controller_state.throttle = 1.0
            agent.controller_state.boost = True
            their_goal = agent.get_opp_goal_pos()
            ball_to_goal_dir = (their_goal - ball_pos).normalize()

            if not self._is_diagonal or self._speed_flipped:
                offset_dist = 170 if self._speed_flipped else 2600
                aim_pos = ball_pos - ball_to_goal_dir * offset_dist
                cubic_aim_at(agent, aim_pos)
                agent.renderer_target = aim_pos
                
                if not self._is_diagonal and not self._speed_flipped:
                    agent.controller_state.steer *= 0.4
            elif agent.me.velocity.magnitude() > 500:
                aim_pos = ball_pos
                cubic_aim_at(agent, aim_pos)
                agent.renderer_target = aim_pos

            if packet.match_info.match_phase != MatchPhase.Kickoff:
                agent.maneuver_lock = 0.0
                agent._kickoff_completed = True
            elif agent.me.velocity.magnitude() > (600 if self._is_diagonal else 700 + abs(agent.me.location.x) * 3) and not self._speed_flipped:
                self._speed_flipped = True
                offset_dist = 250 if self._is_diagonal else -1000
                aim_dir = ball_pos - ball_to_goal_dir * offset_dist
                flat_dir = vec_flat_direction(agent.me.location, aim_dir)
                self._speed_flip = SpeedFlip(flat_dir)
            elif agent.me.location.dist(ball_pos) < 800 and self._time_on_ground > 0.1:
                hit_dir = (their_goal - ball_pos).normalize()
                if self.type == 'central':
                    agent._front_dodge_direction = hit_dir
                    agent._front_dodge_jump_time = 0.18
                    agent._front_dodge_start_time = -1.0
                    agent._front_dodge_step = 0
                    agent._front_dodge_input = None
                    agent.maneuver = Maneuver_FrontDodge
                else:
                    local_hit = agent.me.local(hit_dir)
                    agent.flip_dir = Vector(x=-local_hit.x, y=local_hit.y, z=0)
                    agent.flip_throttle = 1.0
                    agent.maneuver = Maneuver_Flip
                agent.maneuver_lock = 0.8 if self.type == 'speed' else 1.25
                agent._kickoff_completed = True


def Maneuver_Flip(agent, time):
    duration = 0.8
    elapsed = duration - time
    flip_throttle = getattr(agent, "flip_throttle", 1.0)
    agent.controller_state.throttle = flip_throttle

    # Jump 0.05s, release 0.05s, dodge 0.2s.
    if elapsed <= 0.05:
        agent.controller_state.jump = True
    elif elapsed <= 0.10:
        agent.controller_state.jump = False
    elif elapsed <= 0.30: 
        agent.controller_state.jump = True
        agent.controller_state.pitch = agent.flip_dir.x
        agent.controller_state.yaw = agent.flip_dir.y
    else:
        agent.controller_state.jump = False
        agent.controller_state.throttle = flip_throttle


def Maneuver_HalfFlip(agent, time):
    time_elapsed = 1.2 - time
    if time_elapsed < 0.1:
        agent.controller_state.jump = True
        agent.controller_state.pitch = 1.0
    elif time_elapsed < 0.2:
        agent.controller_state.jump = False
        agent.controller_state.pitch = 1.0
    elif time_elapsed < 0.7:
        agent.controller_state.jump = True
        agent.controller_state.pitch = -1.0
        agent.controller_state.roll = 1.0
    elif time_elapsed < 1.0:     
        defaultPD(agent, agent.me.local(agent.me.forward), up=agent.me.local(Vector(z=1)))
        agent.controller_state.handbrake = True 
    else:
        agent.maneuver_lock = 0.0

class AzureRL(Bot):
    def initialize(self):
        self.controller_state = ControllerState()
        self.kickoff_manager = None
        self._kickoff_completed = False
        self.maneuver = None
        self.maneuver_lock = 0.0
        self.flip_dir = Vector()
        self.flip_throttle = 1.0
        self.drive_backwards = False
        self.delta = 1 / 120.0
        self.last_aerial_end_time = -999.0
        self.AERIAL_COOLDOWN = 1.0
        self.JUMP_COOLDOWN = 0.7
        self.last_ground_time = 0.0   # Track when bot was last on ground
        self.last_landing_time = -999.0  # Set once at the moment of touchdown
        self._was_airborne = False        # Previous-tick airborne state
        self.quickchat_cooldown = 1.5
        self.last_quickchat_time = {
            "save": -999.0,
            "aerial": -999.0,
        }
  
        self.time = 0.0
        self.prev_time = 0.0   
        self.me = CarObject(0) 
        self.ball = BallObject()
        self.teammates: list[CarObject] = []
        self.opponents: list[CarObject] = []
        self.renderer_target = None
        self.mode = "Idle"
        self.cached_shot_target = Vector()
        self.logger.info(f"AzureRL initialized! Player ID: {self.player_id}, Team: {self.team}")

    def get_values(self, packet: GamePacket):
        self.time = packet.match_info.seconds_elapsed
        # Packet timestamps are not guaranteed to advance every callback. A
        # repeated timestamp used to freeze maneuver locks for that frame,
        # while a large jump (lag/replay seek) could expire a flip or kickoff
        # maneuver immediately. Keep the normal 120 Hz delta unchanged, but
        # bound anomalous values so state machines continue predictably.
        raw_delta = self.time - self.prev_time
        self.delta = clamp(raw_delta, 1 / 240.0, 1 / 30.0)
        self.prev_time = self.time

        my_car_index = -1
        for i in range(len(packet.players)):
            if packet.players[i].player_id == self.player_id:
                my_car_index = i
                break

        if my_car_index == -1:
            self.logger.warn("Could not find self in packet.players!")
            return 

        self.me.update(packet, car_index=my_car_index)
        self.ball.update(packet)
        
        if not self.me.airborne:
            self.last_ground_time = self.time
            if self._was_airborne:
                self.last_landing_time = self.time
        self._was_airborne = self.me.airborne
        
        teammate_indices = [i for i in range(len(packet.players)) if packet.players[i].team == self.team and i != my_car_index]
        opponent_indices = [i for i in range(len(packet.players)) if packet.players[i].team != self.team]

        # Resize lists only when player count changes (avoids allocations at 120Hz)
        while len(self.teammates) < len(teammate_indices):
            self.teammates.append(CarObject(0))
        del self.teammates[len(teammate_indices):]

        while len(self.opponents) < len(opponent_indices):
            self.opponents.append(CarObject(0))
        del self.opponents[len(opponent_indices):]

        for obj, idx in zip(self.teammates, teammate_indices):
            obj.update(packet, car_index=idx)
        for obj, idx in zip(self.opponents, opponent_indices):
            obj.update(packet, car_index=idx)

    def send_quickchat(self, key: str, message: str):
        last_time = self.last_quickchat_time.get(key, -999.0)
        if self.time - last_time < self.quickchat_cooldown:
            return
        self.last_quickchat_time[key] = self.time
        self.send_match_comm(b"", message)

    def _big_boost_reserved_by_teammate(self, packet: GamePacket, pad_pos: Vector, my_dist_to_pad: float) -> bool:
        # Keep every large boost pad exclusive by assigning it to the closest teammate.
        BOOST_RESERVATION_TOLERANCE = 10.0
        for player in packet.players:
            if player.team != self.team or player.player_id == self.player_id:
                continue
            if player.demolished_timeout > 0:
                continue

            teammate_pos = pos(player)
            teammate_dist = teammate_pos.dist(pad_pos)
            teammate_is_closer = teammate_dist < my_dist_to_pad - BOOST_RESERVATION_TOLERANCE
            teammate_wins_tie = (
                abs(teammate_dist - my_dist_to_pad) <= BOOST_RESERVATION_TOLERANCE and
                player.player_id < self.player_id
            )
            if teammate_is_closer or teammate_wins_tie:
                return True
        return False

    def try_front_flip_to_target(self, target_pos: Vector, yaw_angle: float):
        if self.maneuver_lock > 0 or self.me.airborne or self.me.jumped or self.me.doublejumped:
            return

        # Must have been grounded long enough to straighten out after the previous flip
        MIN_GROUND_TIME = 0.2
        if self.time - self.last_landing_time < MIN_GROUND_TIME:
            return

        target_flat = Vector(target_pos.x, target_pos.y, self.me.location.z)
        dist_to_target = self.me.location.dist(target_flat)
        if dist_to_target < 3200:
            return

        if self.me.location.z > 105:
            return

        surface_normal = self.get_surface_normal(self.me.location)
        if surface_normal.z < 0.99:
            return

        aligned = abs(yaw_angle) < 0.15
        moving_forward = self.me.velocity.dot(self.me.forward) > 1000
        not_turning_around = self.me.local(target_pos - self.me.location).x > 0
        if not (aligned and moving_forward and not_turning_around):
            return

        self.flip_dir = Vector(x=-1.0, y=0.0, z=0.0)
        self.flip_throttle = 1.0
        self.maneuver = Maneuver_Flip
        self.maneuver_lock = 0.8
    
    def _run_maneuver(self, packet: GamePacket):
        if self.maneuver in (Maneuver_Flip, Maneuver_HalfFlip):
            self.maneuver(self, self.maneuver_lock)
        else:
            self.maneuver(self, packet)

    def brain(self, packet: GamePacket):
        if self.me.demolished:
            return 

        if packet.match_info.match_phase == MatchPhase.Kickoff:
            self.handle_kickoff(packet)
            
            # Tick the maneuver immediately so the kickoff taker doesn't waste frame 1
            if self.maneuver_lock > 0 and self.maneuver:
                self._run_maneuver(packet)
                
            # Unconditionally return so non-kickoff takers don't overwrite go_for_kickoff_boost
            return

        # Pre-calculate best opening once per frame
        self.cached_shot_target = self.choose_shot_target()

        # 1. Emergency defense check
        if self.should_return_to_defense_emergency():
            self.emergency_return_to_defense()
            return

        # 2. Saves — absolute priority
        save_opportunity = self.find_save_opportunity()
        
        if save_opportunity:
            self.execute_shot(save_opportunity)
            return

        # 3. Offensive shot
        shot_opportunity = self.find_best_shot_opportunity()

        if shot_opportunity and self.should_i_commit(shot_opportunity.get('ball_at_intercept', self.ball.location), shot_opportunity['intercept_time']):
            self.execute_shot(shot_opportunity)
            return
            
        # 4. Boost collection — only when truly needed
        BOOST_LOW_THRESHOLD = 30
        BOOST_TARGET = 70
        dist_to_ball = self.me.location.dist(self.ball.location)
        should_collect_boost = (self.me.boost < BOOST_LOW_THRESHOLD) or \
                              (self.me.boost < BOOST_TARGET and self.is_ball_safe() and dist_to_ball > 4000)
        
        if should_collect_boost:
            self.go_for_boost(packet)
            return  

        self.shadow_ball()
    
    def should_return_to_defense_emergency(self) -> bool:
        """Check if we should emergency return to defensive zone."""
        own_goal_pos = self.get_own_goal_pos()
        ball_pos = self.ball.location
        
        is_in_offensive_half = (self.team == 0 and self.me.location.y > 0) or (self.team == 1 and self.me.location.y < 0)
        
        if not is_in_offensive_half:
            return False
        
        ball_in_defensive_half = (self.team == 0 and ball_pos.y < 0) or (self.team == 1 and ball_pos.y > 0)
        
        if not ball_in_defensive_half:
            return False
        
        dist_to_defensive_zone = abs(self.me.location.y - own_goal_pos.y)
        
        # Emergency if: ball in defensive half, we're far from goal, and an opponent has better position
        if dist_to_defensive_zone > 3000:
            for opp in self.opponents:
                if not opp.demolished:
                    opp_dist_to_goal = opp.location.dist(own_goal_pos)
                    my_dist_to_goal = self.me.location.dist(own_goal_pos)
                    if opp_dist_to_goal < my_dist_to_goal - 1000:
                        return True  # Emergency return needed
        
        return False
    
    def emergency_return_to_defense(self):
        """Emergency boost to return between the ball and our goal."""
        self.mode = "Defend"
        own_goal_pos = self.get_own_goal_pos()
        ball_to_goal = (own_goal_pos - self.ball.location).normalize()
        ball_goal_distance = own_goal_pos.dist(self.ball.location)
        shadow_distance = clamp(ball_goal_distance * 0.45, 1200, 2500)
        shadow_distance = min(shadow_distance, max(0.0, ball_goal_distance - 450.0))
        defensive_target = self.ball.location + ball_to_goal * shadow_distance
        drive_to_target(self, defensive_target, target_speed=2300, boost=True)

    def get_ball_prediction(self):
        return self.ball_prediction

    def get_own_goal_pos(self) -> Vector:
        return Vector(0, -FIELD_LENGTH / 2 if self.team == 0 else FIELD_LENGTH / 2, 0)
        
    def get_opp_goal_pos(self) -> Vector:
        return Vector(0, FIELD_LENGTH / 2 if self.team == 0 else -FIELD_LENGTH / 2, 0)

    def is_ball_safe(self) -> bool:
        own_goal_y = self.get_own_goal_pos().y 
        return (self.ball.location.y * own_goal_y) < 0

    def is_in_defensive_area(self, position: Vector = None) -> bool:
        """Check if position (default: bot's position) is in defensive third."""
        if position is None:
            position = self.me.location
        if self.team == 0:
            return position.y < -FIELD_LENGTH / 6
        else:
            return position.y > FIELD_LENGTH / 6

    def estimate_travel_time(self, start: CarObject, target_pos: Vector, avg_speed: float = 1600.0) -> float:
        delta = target_pos - start.location
        dist = delta.magnitude()
        if dist < 10: return 0.0
        
        # Penalize for turning
        vel_dir = start.velocity.normalize() if start.velocity.magnitude() > 100 else start.forward
        angle = vel_dir.angle(delta)
        turn_time = angle * 0.45
        drive_time = dist / avg_speed
        return turn_time + drive_time
    
    def estimate_travel_time_for_save(self, start: CarObject, target_pos: Vector, avg_speed: float = 2300.0) -> float:
        """Like estimate_travel_time, but also considers rolling backwards to
        the target. estimate_travel_time only models turning to face the
        target and driving forward, so when the car's back is already facing
        the ball it hugely overestimates the time needed (a full turn-around)
        even though reversing straight into it is much faster. Ground saves
        should take whichever is quicker."""
        delta = target_pos - start.location
        dist = delta.magnitude()
        if dist < 10:
            return 0.0

        forward_angle = start.forward.angle(delta)
        forward_time = forward_angle * 0.45 + dist / avg_speed

        REVERSE_SPEED = 1300.0  # cars reverse slower than they drive forward
        back_angle = math.pi - forward_angle
        reverse_time = back_angle * 0.45 + dist / REVERSE_SPEED

        return min(forward_time, reverse_time)

    def _can_start_aerial_takeoff(self, target: Vector = None) -> bool:
        """Whether an aerial can plausibly be started from floor, wall, or air."""
        if self.me.demolished or self.me.doublejumped:
            return False
        if self.time - self.last_landing_time < 0.2:
            return False
        
        if self.me.boost < 8:
            if target is not None:
                ball_height = target.z
                horizontal_dist = Vector(target.x - self.me.location.x, target.y - self.me.location.y, 0).magnitude()
                if 120 <= ball_height <= 300 and horizontal_dist < 400:
                    return True
            return False

        speed = self.me.velocity.magnitude()

        if not self.me.airborne and speed < 900:
            return False

        if self.me.airborne:
            return True

        surface_normal = self.get_surface_normal(self.me.location)
        on_flat_ground = surface_normal.z > 0.85

        if on_flat_ground:
            return (not self.me.jumped) and self.me.location.z < 320

        # Wall or backboard: allow takeoff from any height.
        return True

    def get_surface_normal(self, location: Vector) -> Vector:
        """Returns the normal vector of the surface at the given location (floor or walls)."""
        if abs(location.x) > 3800:
            return Vector(-1 if location.x > 0 else 1, 0, 0)
        if abs(location.y) > 4750:
            return Vector(0, -1 if location.y > 0 else 1, 0)
        return Vector(0, 0, 1)

    def is_ball_on_wall(self, ball_pos: Vector) -> bool:
        """Determines if the ball is in a position reachable by driving up a wall."""
        on_side = abs(ball_pos.x) > 3400 and 200 < ball_pos.z < 2500
        on_back = abs(ball_pos.y) > 5000 and 200 < ball_pos.z < 2500
        return on_side or on_back

    def is_towards_own_goal(self, start_pos: Vector, direction: Vector) -> bool:
        """Checks if a ray from start_pos in direction points into the bot's own goal."""
        own_goal = self.get_own_goal_pos()
        if abs(direction.y) < 1e-3: return False
        
        # If direction points away from our goal half, it's safe
        if (self.team == 0 and direction.y > 0) or (self.team == 1 and direction.y < 0):
            return False
            
        # Time to reach our goal line
        time = (own_goal.y - start_pos.y) / direction.y
        if time < 0: return False
        
        # Where it hits the goal plane
        intercept_x = start_pos.x + direction.x * time
        intercept_z = start_pos.z + direction.z * time
        
        return abs(intercept_x) < (GOAL_WIDTH / 2 + 100) and 0 <= intercept_z <= (GOAL_HEIGHT + 100)


    def can_reach_intercept(self, ball_pos: Vector, intercept_time: float,
                            allow_jump=True, allow_aerial=True) -> bool:
        tti = intercept_time - self.time
        if tti <= 0.05:
            return False

        target = ball_pos
        flat_dist = (target - self.me.location).flatten().magnitude()
        current_speed = max(0.0, self.me.velocity.dot(self.me.forward))

        # Simple forward simulation
        reachable_speed = min(2300.0, current_speed + 1000.0 * tti)
        reachable_dist = reachable_speed * tti

        # Turning penalty
        angle = self.me.forward.angle((target - self.me.location).flatten())
        reachable_dist -= angle * 400.0

        # Height capability
        max_height = 120.0
        if allow_jump and not self.me.doublejumped:
            max_height = 500.0
        if allow_aerial and self.me.boost > 10:
            max_height = 10000.0

        return (
            flat_dist <= reachable_dist + 150 and
            target.z <= max_height
        )

    def find_save_opportunity(self):
        prediction = self.get_ball_prediction()
        if prediction is None:
            return None
        
        own_goal_pos = self.get_own_goal_pos()
        ball_pos = self.ball.location
        
        # Check if ball is in our defensive half or moving toward our goal
        ball_moving_to_goal = (self.ball.velocity.y * (own_goal_pos.y - ball_pos.y)) > 0
        ball_in_defensive_half = (self.team == 0 and ball_pos.y < 0) or (self.team == 1 and ball_pos.y > 0)
        
        # Emergency: ball very close to goal and heading in
        ball_close_to_goal = abs(ball_pos.y - own_goal_pos.y) < 2500
        ball_in_goal_width = abs(ball_pos.x) < GOAL_WIDTH / 2 + 500
        
        is_emergency = ball_close_to_goal and ball_in_goal_width and ball_moving_to_goal
        
        # Opponent in shooting position (close to ball in our defensive half)
        opponent_threat = False
        for opp in self.opponents:
            if not opp.demolished:
                opp_dist_to_ball = opp.location.dist(ball_pos)
                if opp_dist_to_ball < 800 and ball_in_defensive_half:
                    opponent_threat = True
                    break
        
        should_save = is_emergency or (opponent_threat and ball_moving_to_goal)
        
        if not should_save:
            return None
        
        best_save = None
        earliest_intercept = 999.0

        # Calculate best clear target away from our goal — constant across the
        # whole scan, so it's computed once instead of on every slice.
        safe_clear_target = self.get_safe_clear_target()

        # Biggest score discount any candidate below can receive (emergency
        # ground/aerial -1.0, plus the -0.25 aerial bonus). Used to bound how
        # far ahead a later, cheaper-scoring slice could still beat the best
        # candidate found so far, so the scan can stop as soon as that's
        # provably impossible.
        MAX_SAVE_SCORE_DISCOUNT = 1.25

        # Look for intercept points across the full ~5s prediction: dense
        # sampling for the first 2s, sparser beyond that.
        for i in adaptive_slice_indices(len(prediction.slices), near_limit=240, near_step=2, far_step=5):
            ball_phys = prediction.slices[i]
            ball_pred_pos = pos(ball_phys)
            intercept_time = ball_phys.game_seconds
            time_to_intercept = intercept_time - self.time

            if time_to_intercept < 0.1:
                continue
            if time_to_intercept > 5.0:
                # Slices are in increasing time order, so nothing further out
                # will be in range either.
                break

            # Once even the best-case discounted score of this (and every
            # later, since time-ordered) slice can't beat what's already
            # found, further scanning can't help.
            if best_save is not None and (time_to_intercept - MAX_SAVE_SCORE_DISCOUNT) > earliest_intercept:
                break

            # Skip if ball is past our goal line
            if abs(ball_pred_pos.y) > FIELD_LENGTH/2 + 100:
                continue

            if not self.can_reach_intercept(ball_pred_pos, intercept_time):
                continue

            ball_to_target = (safe_clear_target - ball_pred_pos).normalize()
            approach_target = ball_pred_pos - ball_to_target * 50
            
            save_score = time_to_intercept
            if is_emergency:
                save_score -= 1.0

            # Aerial save for high or wall-height threats
            is_wall_ball = self.is_ball_on_wall(ball_pred_pos)
            can_attempt_aerial = (self.time - self.last_aerial_end_time) >= self.AERIAL_COOLDOWN
            aerial_height_or_wall = ball_pred_pos.z > 150 or is_wall_ball
            is_first_jump_ball = ball_pred_pos.z <= 300 and not self.me.airborne
            if (can_attempt_aerial and aerial_height_or_wall and self._can_start_aerial_takeoff(ball_pred_pos) and
                    (self.me.boost >= 12 or (is_first_jump_ball and self.me.boost >= 0)) and 0.25 < time_to_intercept < 4.0):
                target_vec = ball_pred_pos - self.me.location
                angle_to_target = self.me.forward.angle(target_vec)
                aerial_travel_time = self.estimate_travel_time(self.me, ball_pred_pos, 1800.0)
                travel_fraction = aerial_travel_time / time_to_intercept if time_to_intercept > 0 else 1.0

                if angle_to_target < math.pi / 1.1 and travel_fraction <= 1.0:
                    aerial_score = save_score - 0.25
                    if aerial_score < earliest_intercept:
                        earliest_intercept = aerial_score
                        best_save = {
                            "type": "aerial",
                            "intercept_time": intercept_time,
                            "ball_at_intercept": ball_pred_pos,
                            "shot_target": safe_clear_target,
                            "urgent": is_emergency
                        }
            
            # Ground save — account for reversing straight into the ball when
            # our back is already facing it, instead of assuming we always
            # have to turn around and drive forward.
            my_travel_time = self.estimate_travel_time_for_save(self.me, ball_pred_pos, 2300.0)
            if my_travel_time > time_to_intercept:
                continue
            
            if save_score < earliest_intercept:
                earliest_intercept = save_score
                best_save = {
                    "type": "ground",
                    "intercept_time": intercept_time,
                    "approach_target": approach_target,
                    "ball_at_intercept": ball_pred_pos,
                    "shot_target": safe_clear_target,
                    "urgent": is_emergency
                }
        
        return best_save
    
    def get_safe_clear_target(self) -> Vector:
        """Returns a clear target identical to the shooting target."""
        opp_goal_pos = self.get_opp_goal_pos()
        return Vector(0, opp_goal_pos.y, 550.0)
        
    def find_best_shot_opportunity(self):
        prediction = self.get_ball_prediction()
        if prediction is None:
            return None
        
        best_shot = None
        earliest_time = 6.0 
        shot_target = self.cached_shot_target
        can_attempt_aerial = (self.time - self.last_aerial_end_time) >= self.AERIAL_COOLDOWN

        # Scan the full ~5s prediction instead of stopping at 4s: dense
        # sampling for the first 2s, sparser beyond that. Scores here are
        # plain time_to_intercept (no discount), and slices are visited in
        # increasing time order, so the first accepted candidate is already
        # the earliest possible one — each acceptance branch below breaks out
        # immediately instead of scanning the rest of the prediction.
        max_slices = len(prediction.slices)
        for i in adaptive_slice_indices(max_slices, near_limit=240, near_step=2, far_step=5):
            ball_predict_phys = prediction.slices[i]
            ball_predict_pos = pos(ball_predict_phys)
            intercept_time = ball_predict_phys.game_seconds
            time_to_intercept = intercept_time - self.time           
            
            if time_to_intercept < 0.05:
                continue 
            
            if not self.can_reach_intercept(ball_predict_pos, intercept_time):
                continue

            if can_attempt_aerial and ball_predict_pos.z > 220:
                is_first_jump_ball = ball_predict_pos.z <= 300 and self.me.boost < 15 and not self.me.airborne
                if self._can_start_aerial_takeoff(ball_predict_pos) and (self.me.boost >= 15 or is_first_jump_ball):
                    target_vec = ball_predict_pos - self.me.location
                    angle_to_target = self.me.forward.angle(target_vec)
                    my_travel_time = self.estimate_travel_time(self.me, ball_predict_pos, 1800.0)
                    travel_fraction = my_travel_time / time_to_intercept if time_to_intercept > 0 else 1.0

                    if angle_to_target < AERIAL_START_ANGLE and travel_fraction < 0.96 and time_to_intercept < earliest_time:
                        earliest_time = time_to_intercept
                        best_shot = {
                            "type": "aerial",
                            "intercept_time": intercept_time,
                            "shot_target": shot_target,
                            "ball_at_intercept": ball_predict_pos
                        }
                        break  # time-ordered scan: this is already the earliest possible shot
            
            # Ground / wall shot
            is_wall_ball = self.is_ball_on_wall(ball_predict_pos)
            if ball_predict_pos.z < 850 or is_wall_ball:                
                aim_vec = (shot_target - ball_predict_pos).normalize()
                
                if is_wall_ball:
                    is_first_jump_ball = ball_predict_pos.z <= 300 and self.me.boost < 15 and not self.me.airborne
                    if can_attempt_aerial and self._can_start_aerial_takeoff(ball_predict_pos) and (self.me.boost >= 15 or is_first_jump_ball):
                        target_vec = ball_predict_pos - self.me.location
                        angle_to_target = self.me.forward.angle(target_vec)
                        my_travel_time = self.estimate_travel_time(self.me, ball_predict_pos, 1800.0)
                        travel_fraction = my_travel_time / time_to_intercept if time_to_intercept > 0 else 1.0

                        if angle_to_target < AERIAL_START_ANGLE and travel_fraction < 0.96 and time_to_intercept < earliest_time:
                            earliest_time = time_to_intercept
                            best_shot = {
                                "type": "aerial",
                                "intercept_time": intercept_time,
                                "shot_target": shot_target,
                                "ball_at_intercept": ball_predict_pos
                            }
                            break  # time-ordered scan: this is already the earliest possible shot
                    
                    # Use wall driving approach
                    wall_normal = self.get_surface_normal(ball_predict_pos)
                    approach_target = ball_predict_pos - wall_normal * 100
                    travel_speed = 2100.0
                else:
                    approach_target = ball_predict_pos.flatten() - aim_vec.flatten() * 120 
                    travel_speed = 2000.0
                
                my_travel_time = self.estimate_travel_time(self.me, approach_target, travel_speed)
                
                if is_wall_ball:
                    jump_prep_time = 0.0
                else:
                    MAX_REACHABLE_HEIGHT = 300
                    if ball_predict_pos.z > MAX_REACHABLE_HEIGHT:
                        jump_prep_time = 999
                    else:
                        jump_prep_time = 0.0 if ball_predict_pos.z < 80 else (ball_predict_pos.z - 80) / 1200.0

                if (my_travel_time + jump_prep_time) < time_to_intercept and time_to_intercept < earliest_time:
                    earliest_time = time_to_intercept
                    best_shot = {
                        "type": "ground",
                        "intercept_time": intercept_time,
                        "approach_target": approach_target,
                        "ball_at_intercept": ball_predict_pos,
                        "shot_target": shot_target
                    }
                    break  # time-ordered scan: this is already the earliest possible shot
        return best_shot

    def should_i_commit(self, ball_pos: Vector, my_intercept_time: float, is_save: bool = False) -> bool:
        my_relative_time = my_intercept_time - self.time
        
        # Check if ball is in defensive area
        ball_in_defensive = self.is_in_defensive_area(ball_pos)
        
        # More aggressive buffer for saves; back off only if teammate is much closer
        buffer = 0.5 if is_save else -0.2
        
        # In defensive area, be more aggressive about defending
        if ball_in_defensive and not is_save:
            buffer = -0.1
        
        my_dist_to_ball = self.me.location.dist(ball_pos)

        for tm in self.teammates:
            if tm.demolished:
                continue
            
            tm_dist_to_ball = tm.location.dist(ball_pos)
            
            if ball_in_defensive and not is_save:
                # Only back off if teammate is significantly closer (500 units)
                if tm_dist_to_ball < my_dist_to_ball - 500:
                    return False
            else:
                if tm_dist_to_ball < my_dist_to_ball - 300:
                    return False
                 
            tm_relative_time = self.estimate_travel_time(tm, ball_pos, 1500.0)
            
            if tm_relative_time < my_relative_time - buffer:
                return False       
        return True 
        
    def choose_shot_target(self) -> Vector:
        """Pick a goal target that is harder to block than the goal center.

        The center is a useful default, but using it for every shot makes the
        bot very predictable. Prefer the far post when the ball is wide, then
        use the candidate with the most separation from the nearest defender.
        Keeping the candidates inside the posts also makes this safe for
        ground shots and aerials that reuse ``cached_shot_target``.
        """
        opp_goal_pos = self.get_opp_goal_pos()
        goal_y = opp_goal_pos.y
        candidates = (
            Vector(-700.0, goal_y, 550.0),
            Vector(0.0, goal_y, 550.0),
            Vector(700.0, goal_y, 550.0),
        )

        ball_x = self.ball.location.x
        wide_ball = abs(ball_x) > 350.0
        far_post_x = -math.copysign(700.0, ball_x) if wide_ball else 0.0

        active_opponents = [opp for opp in self.opponents if not opp.demolished]
        best_target = candidates[1]
        best_score = float("-inf")
        for candidate in candidates:
            nearest_defender = min(
                (opp.location.dist(candidate) for opp in active_opponents),
                default=2500.0,
            )
            far_post_bonus = 450.0 if candidate.x == far_post_x else 0.0
            center_bonus = 50.0 if candidate.x == 0.0 and not wide_ball else 0.0
            center_penalty = 100.0 if candidate.x == 0.0 and wide_ball else 0.0
            score = nearest_defender * 0.35 + far_post_bonus + center_bonus - center_penalty
            if score > best_score:
                best_score = score
                best_target = candidate

        return best_target
        
    def handle_kickoff(self, packet):
        ball_pos = pos(packet.balls[0]) if packet.balls else Vector(0, 0, 0)

        # Build a list of (distance, player_id) for all teammates (including self)
        # using the SAME data source (packet) for consistency across all bot instances.
        team_entries = []
        for p in packet.players:
            if p.team != self.team:
                continue
            d = pos(p).dist(ball_pos)
            team_entries.append((d, p.player_id))

        if not team_entries:
            return

        # Find the minimum distance among all teammates
        min_dist = min(d for d, _ in team_entries)

        # Among all teammates within tolerance of that minimum distance,
        # pick the one with the lowest player_id as the kickoff taker.
        KICKOFF_DIST_TOLERANCE = 10.0
        candidates = [(d, pid) for d, pid in team_entries if d <= min_dist + KICKOFF_DIST_TOLERANCE]
        taker_id = min(pid for _, pid in candidates)

        i_am_kickoff_taker = (self.player_id == taker_id)

        if i_am_kickoff_taker:
            self.mode = "Kickoff"
            if not getattr(self, '_kickoff_completed', False):
                if self.kickoff_manager is None:
                    # Re-detect spawn style every kickoff from current position
                    # (the match-start cache can assign wrong type on subsequent kickoffs).
                    ax = abs(self.me.location.x)
                    style = 'central' if ax < KICKOFF_CENTRAL_X else 'speed'
                    self.kickoff_manager = Kickoff(self, packet, kickoff_type=style)
                self.maneuver = self.kickoff_manager.get_output
                self.maneuver_lock = 2.6
            else:
                self.maneuver_lock = 0.0
                self.kickoff_manager = None
        else:
            self.mode = "KickoffBoost"
            self.go_for_kickoff_boost(packet)
            self.maneuver = None
            self.kickoff_manager = None

    def go_for_kickoff_boost(self, packet):
        best_pad = None
        my_goal_y_sign = -1 if self.team == 0 else 1
        boost_pads = self.field_info.boost_pads
        
        # Create a list of available boost pads with their distances
        available_pads = []
        for i, pad in enumerate(boost_pads):
            pad_pos = pos(pad)
            if pad.is_full_boost and pad_pos is not None and (pad_pos.y * my_goal_y_sign) >= 0:
                dist = self.me.location.dist(pad_pos)
                available_pads.append((pad, dist))
        
        # Sort pads by distance (closest first)
        available_pads.sort(key=lambda x: x[1])
        
        # Find the best pad that isn't reserved by a closer teammate
        for pad, dist in available_pads:
            pad_pos = pos(pad)
            if not self._big_boost_reserved_by_teammate(packet, pad_pos, dist):
                best_pad = pad
                break
        
        if best_pad is not None:
            drive_to_target(self, pos(best_pad), target_speed=2300, boost=True, reverse_distance=3000.0, ramp_down=True)
        else:
            drive_to_target(self, self.get_own_goal_pos() * 0.2, target_speed=1400)

    def execute_shot(self, shot_opportunity):
        is_save = shot_opportunity.get('urgent', False)
        if shot_opportunity['type'] == 'aerial':
            self.mode = "Aerial Save" if is_save else "Aerial"
        else:
            self.mode = "Save" if is_save else "Shot"
        if shot_opportunity['type'] == 'ground':
            target_pos = shot_opportunity['approach_target']
            ball_at_intercept = shot_opportunity['ball_at_intercept']
            shot_target = shot_opportunity['shot_target']
            time_to_intercept = shot_opportunity['intercept_time'] - self.time
            dist_to_target = self.me.location.dist(target_pos)
            
            is_save = shot_opportunity.get('urgent', False)
            
            target_height = ball_at_intercept.z
            
            if is_save:
                #self.send_quickchat("save", "Clear it ASAP!")
                if time_to_intercept > 0.1:
                    required_speed = dist_to_target / time_to_intercept
                    target_speed = clamp(required_speed, 0, 2300)
                else:
                    target_speed = 2300
                
                use_boost = True
                drive_to_target(self, target_pos, target_speed=target_speed, boost=use_boost)
                
                # Flip into the ball for saves when close enough
                if dist_to_target < 350 and time_to_intercept < 0.2 and target_height < 200:
                    car_to_ball_vec = (ball_at_intercept - self.me.location).normalize()
                    own_goal = self.get_own_goal_pos()
                    ball_to_goal = (own_goal - ball_at_intercept).normalize()
                    flip_alignment = car_to_ball_vec.dot(ball_to_goal)

                    # Only refuse the flip if it is aimed almost perfectly into our own goal.
                    if flip_alignment <= 0.7:
                        local_flip_target = self.me.local(car_to_ball_vec)
                        self.flip_throttle = -1.0 if local_flip_target.x < 0.0 else 1.0
                        self.flip_dir = Vector(x=-local_flip_target.x, y=local_flip_target.y, z=0)
                        self.maneuver = Maneuver_Flip
                        self.maneuver_lock = 0.8
                        return
            else:
                if time_to_intercept > 0.1:
                    required_speed = dist_to_target / time_to_intercept
                    target_speed = clamp(required_speed, 0, 2300)
                else:
                    target_speed = 2300
                
                drive_to_target(self, target_pos, target_speed=target_speed, boost=True)
            
            # 3. Jump timing — use can_reach_intercept for ground reachability
            ground_reachable = self.can_reach_intercept(
                ball_at_intercept,
                shot_opportunity['intercept_time'],
                allow_aerial=False
            )
            if ground_reachable:
                time_to_reach_z = 0.30 if target_height >= 150 else (0.15 if target_height >= 80 else 0.0)
            else:
                time_to_reach_z = 999

            # 4. Trigger jump; block during cooldown
            time_since_ground = self.time - self.last_ground_time
            jump_cooldown_over = is_save or (time_since_ground >= self.JUMP_COOLDOWN)
            is_wall_ball = self.is_ball_on_wall(ball_at_intercept)
            hit_direction = (ball_at_intercept - self.me.location).normalize()
            dangerous_shot = self.is_towards_own_goal(ball_at_intercept, hit_direction)

            if not jump_cooldown_over:
                time_to_reach_z = 999

            if is_wall_ball:
                wall_normal = self.get_surface_normal(ball_at_intercept)
                wall_entry_point = ball_at_intercept - wall_normal * 100
                
                my_surface_normal = self.get_surface_normal(self.me.location)
                on_wall = abs(my_surface_normal.z) < 0.9
                
                if on_wall or self.me.location.z > 100:
                    # Already on wall — aim behind ball for contact
                    ball_to_target = (shot_target - ball_at_intercept).normalize()
                    scoring_target = ball_at_intercept - ball_to_target * 60
                    
                    target_pos = scoring_target
                    target_speed = min(2300, dist_to_target / time_to_intercept) if time_to_intercept > 0.1 else 2300
                else:
                    # Approaching wall — aim for entry point
                    target_pos = wall_entry_point
                    target_speed = min(2300, dist_to_target / time_to_intercept) if time_to_intercept > 0.1 else 2300
                
                drive_to_target(self, target_pos, target_speed=target_speed, boost=True)
                return

            # 5. Double jump for mid-height balls.
            # Use a wider window for saves so the bot has time to jump while rising.
            if self.me.jumped and not self.me.doublejumped and target_height > 150:
                double_jump_window = 0.8 if is_save else 0.4
                if time_to_intercept < double_jump_window and (is_save or not dangerous_shot):
                    self.controller_state.jump = True
                    
            # 6. Flip/dodge logic
            MAX_FLIP_HEIGHT = 200
            if dist_to_target < 400 and time_to_intercept < 0.3 and target_height < MAX_FLIP_HEIGHT:
                if is_save or not dangerous_shot:
                    car_to_ball_vec = (ball_at_intercept - self.me.location).normalize()

                    # For saves, only skip if flip is almost perfectly aimed at our goal (>0.7).
                    # For offensive shots, use a more conservative threshold (>0.3).
                    own_goal = self.get_own_goal_pos()
                    ball_to_goal = (own_goal - ball_at_intercept).normalize()
                    flip_alignment = car_to_ball_vec.dot(ball_to_goal)
                    alignment_threshold = 0.7 if is_save else 0.3
                    if flip_alignment > alignment_threshold:
                        return

                    local_flip_target = self.me.local(car_to_ball_vec)
                    self.flip_throttle = -1.0 if local_flip_target.x < 0.0 else 1.0
                    self.flip_dir = Vector(x=-local_flip_target.x, y=local_flip_target.y, z=0)
                    self.maneuver = Maneuver_Flip
                    self.maneuver_lock = 0.8
        
        elif shot_opportunity['type'] == 'aerial':
            started_aerial = False
            can_attempt_aerial = (self.time - self.last_aerial_end_time) >= self.AERIAL_COOLDOWN
            is_save = shot_opportunity.get('urgent', False)
            min_boost = 12 if is_save else 20
            max_travel_fraction = 0.98 if is_save else 0.95
            intercept_ball = shot_opportunity.get('ball_at_intercept', self.ball.location)

            if self.maneuver_lock <= 0 and not self.me.demolished and not self.me.doublejumped:
                can_start_aerial = self._can_start_aerial_takeoff(intercept_ball)
                is_first_jump_aerial = (self.me.boost < min_boost and 
                                        intercept_ball.z <= 300 and 
                                        not self.me.airborne)
                if can_start_aerial or is_first_jump_aerial:
                    if self.me.boost >= min_boost or is_first_jump_aerial:
                        tti = shot_opportunity['intercept_time'] - self.time
                        if 0.25 < tti < 3.5 and abs(self.me.angular_velocity.magnitude()) < 5.0:
                            travel_time = self.estimate_travel_time(self.me, intercept_ball, 1800.0)
                            travel_fraction = travel_time / tti if tti > 0 else 1.0
                            if travel_fraction < max_travel_fraction:
                                self.maneuver = Maneuver_Aerial(
                                    self, shot_opportunity['intercept_time'], shot_opportunity['shot_target']
                                )
                                self.maneuver_lock = tti + 0.5
                                self.send_quickchat("aerial", "Flying!")
                                started_aerial = True

            if not started_aerial and self.maneuver_lock <= 0:
                fallback_ball = shot_opportunity.get('ball_at_intercept', self.ball.location)
                fallback_tti = shot_opportunity['intercept_time'] - self.time
                fallback_dist = self.me.location.dist(fallback_ball)
                if fallback_tti > 0.08:
                    fallback_speed = clamp(fallback_dist / fallback_tti, 0, 2300)
                else:
                    fallback_speed = 2300
                drive_to_target(self, fallback_ball, target_speed=fallback_speed, boost=True)

    def go_for_boost(self, packet: GamePacket):
        self.mode = "Boost"
        ball_pos = self.ball.location
        own_goal_pos = self.get_own_goal_pos()
        ball_dist_to_goal = ball_pos.dist(own_goal_pos)
        boost_pads = self.field_info.boost_pads
        
        ball_in_right_half = ball_pos.x >= 0
        
        closest_pad_overall = None
        closest_pad_same_half = None
        dist_overall = 99999.0
        dist_same_half = 99999.0
        
        for i, pad in enumerate(boost_pads):
            pad_pos = pos(pad)
            if not (pad.is_full_boost and i < len(packet.boost_pads) and packet.boost_pads[i].is_active):
                continue
            pad_dist_to_goal = pad_pos.dist(own_goal_pos)
            if pad_dist_to_goal > ball_dist_to_goal:
                continue
            
            my_dist_to_pad = self.me.location.dist(pad_pos)
            if self._big_boost_reserved_by_teammate(packet, pad_pos, my_dist_to_pad):
                continue

            pad_in_same_half = (pad_pos.x >= 0) == ball_in_right_half
            
            if my_dist_to_pad < dist_overall:
                dist_overall = my_dist_to_pad
                closest_pad_overall = pad
            
            if pad_in_same_half and my_dist_to_pad < dist_same_half:
                dist_same_half = my_dist_to_pad
                closest_pad_same_half = pad
        
        best_pad = None
        if closest_pad_same_half is not None:
            if closest_pad_overall is not None and dist_overall * 2 <= dist_same_half:
                best_pad = closest_pad_overall
            else:
                best_pad = closest_pad_same_half
        
        if best_pad is None:
            min_dist = 99999.0
            for i, pad in enumerate(boost_pads):
                pad_pos = pos(pad)
                if i < len(packet.boost_pads) and packet.boost_pads[i].is_active:
                    my_dist_to_pad = self.me.location.dist(pad_pos)
                    if pad.is_full_boost and self._big_boost_reserved_by_teammate(packet, pad_pos, my_dist_to_pad):
                        continue
                    if my_dist_to_pad < 1500:
                        if my_dist_to_pad < min_dist:
                            min_dist = my_dist_to_pad
                            best_pad = pad
                            
        if best_pad is not None:
            use_all_boost = best_pad.is_full_boost
            drive_to_target(self, pos(best_pad), target_speed=2300, boost=use_all_boost, ramp_down=True)
        else:
            my_goal = self.get_own_goal_pos()
            safe_pos = my_goal + (self.ball.location - my_goal).normalize() * 2000
            drive_to_target(self, safe_pos, target_speed=2300, boost=True)
            
    def shadow_ball(self):
        self.mode = "Shadow"
        WALL_X_LIMIT = 3700  # Stay within field bounds
        my_goal = self.get_own_goal_pos()
        ball_pos = self.ball.location
        
        ball_in_defensive_half = (self.team == 0 and ball_pos.y < 0) or (self.team == 1 and ball_pos.y > 0)
        
        if ball_in_defensive_half:
            # Position between ball and goal, closer to ball
            ball_to_goal = (my_goal - ball_pos).normalize()
            dist_ball_to_goal = ball_pos.dist(my_goal)
            
            shadow_distance = clamp(dist_ball_to_goal * 0.3, 800, 1500)
            shadow_pos = ball_pos + ball_to_goal * shadow_distance
            
            # Shift laterally to cover shot angles when ball is wide
            lateral_offset = Vector(0, 0, 0)
            if abs(ball_pos.x) > 1000:
                lateral_offset = Vector(-math.copysign(300, ball_pos.x), 0, 0)
            
            shadow_pos = shadow_pos + lateral_offset
        else:
            # Ball in offensive half — standard mid-depth shadow
            dist_ball_to_goal = ball_pos.dist(my_goal)
            shadow_distance = clamp(dist_ball_to_goal * 0.5, 0, 2500)
            shadow_pos = ball_pos + (my_goal - ball_pos).normalize() * shadow_distance
        
        SUFFICIENT_BOOST = 60
        should_conserve_boost = self.me.boost > SUFFICIENT_BOOST
        
        is_wrong_side = (self.me.location.y - ball_pos.y) * my_goal.y < 0
        dist_to_ball = self.me.location.dist(ball_pos)

        # Use Half-Flip if far out of position and traveling in the wrong direction
        if is_wrong_side and dist_to_ball > 3000 and self.me.velocity.dot(self.me.forward) < -500:
             if self.maneuver_lock <= 0:
                 self.maneuver = Maneuver_HalfFlip
                 self.maneuver_lock = 1.2
                 return
        
        if is_wrong_side and dist_to_ball < 2500:
            safe_target_x = math.copysign(min(850, WALL_X_LIMIT), self.me.location.x)
            safe_target_y = my_goal.y * 0.95
            safe_target = Vector(safe_target_x, safe_target_y, 0)
            drive_to_target(self, safe_target, target_speed=2300, boost=True)
            return

        GOAL_LINE_BUFFER = 100.0 
        if self.team == 0:
            shadow_pos.y = max(shadow_pos.y, my_goal.y + GOAL_LINE_BUFFER)
        else:
            shadow_pos.y = min(shadow_pos.y, my_goal.y - GOAL_LINE_BUFFER)

        # Prevent shadowing on side walls (but allow backboard)
        if abs(shadow_pos.x) > WALL_X_LIMIT:
            shadow_pos.x = math.copysign(WALL_X_LIMIT, shadow_pos.x)

        if should_conserve_boost:
            # High boost: slower, more controlled shadowing
            target_speed = clamp(self.ball.velocity.magnitude() + 100, 800, 1800)
            use_boost = False
        else:
            # Low boost: maintain presence, use boost only when far from ball
            target_speed = clamp(self.ball.velocity.magnitude() + 200, 0, 2300)
            use_boost = dist_to_ball > 4000
            
        drive_to_target(self, shadow_pos, target_speed=target_speed, boost=use_boost)

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0:
            return ControllerState()
            
        self.controller_state = ControllerState()
        self.get_values(packet)

        self.renderer.begin_rendering()
        if self.ball_prediction:
            self.renderer.draw_polyline_3d([s.physics.location for s in self.ball_prediction.slices[::20]], self.renderer.cyan)
        
        if self.renderer_target:
            start = Vector3(self.me.location.x, self.me.location.y, self.me.location.z)
            end = Vector3(self.renderer_target.x, self.renderer_target.y, self.renderer_target.z)
            
            self.renderer.draw_line_3d(start, end, self.renderer.red)
        self.renderer.end_rendering()

        # Render the current mode above the car in the bot's team color (debug overlay).
        if self.renderer.can_render:
            team_color = self.renderer.orange if self.team == 1 else self.renderer.blue
            with self.renderer.context(group_id="mode_label", default_color=team_color):
                anchor = CarAnchor(self.index, Vector3(0.0, 0.0, 120.0))
                self.renderer.draw_string_3d(self.mode, anchor, 2.0)
        
        
        if self.maneuver_lock > 0:
            if self.maneuver:
                self._run_maneuver(packet)
            self.maneuver_lock -= self.delta 
        else:
            # Track when aerial maneuver ends for cooldown
            if isinstance(self.maneuver, Maneuver_Aerial):
                self.last_aerial_end_time = self.time
            
            self.maneuver = None
            self.kickoff_manager = None
            # Only reset _kickoff_completed when not in kickoff phase — otherwise the
            # completed flag is cleared on the very next tick and a second kickoff starts.
            if packet.match_info.match_phase != MatchPhase.Kickoff:
                self._kickoff_completed = False
            
            # Recovery: point wheels down and forward when airborne without jumping
            if self.me.airborne and not self.controller_state.jump:
                target_vel = self.me.velocity if self.me.velocity.magnitude() > 500 else self.me.forward
                defaultPD(self, self.me.local(target_vel.flatten()), upside_down=False)
            
            self.brain(packet)
            
        return self.controller_state

if __name__ == "__main__":
    AzureRL().run()
