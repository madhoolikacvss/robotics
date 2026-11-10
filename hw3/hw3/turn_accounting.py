"""Diagnostic-only accounting for each continuous, same-direction turn."""
import math


FIELDS = [
    'turn_id', 'event', 'reason', 'start_sec', 'stop_sec', 'summary_sec',
    'direction', 'command_duration_sec', 'command_min_abs', 'command_max_abs',
    'start_yaw_rad', 'end_yaw_rad', 'predicted_yaw_rad',
    'tag_correction_yaw_rad', 'post_stop_tag_correction_yaw_rad',
    'accounted_yaw_rad', 'wrapped_balance_error_rad',
    'accepted_updates', 'rejected_updates', 'initialized_tags',
    'post_stop_accepted_updates', 'prediction_steps', 'prediction_dt_sum',
    'prediction_dt_squared_sum', 'angular_process_variance_added',
    'start_yaw_variance', 'end_yaw_variance',
]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TurnAccounting:
    def __init__(self, emit, settle_seconds=1.5):
        self.emit = emit
        self.settle_seconds = settle_seconds
        self.turn = None
        self.counter = 0

    def snapshot(self, now, yaw, variance, event, reason):
        t = self.turn
        accounted = (t['predicted_yaw_rad'] + t['tag_correction_yaw_rad']
                     + t['post_stop_tag_correction_yaw_rad'])
        self.emit(dict(t, event=event, reason=reason, summary_sec=now,
                       command_duration_sec=(t['stop_sec'] if t['stop_sec'] is not None
                                             else now) - t['start_sec'],
                       end_yaw_rad=yaw, end_yaw_variance=variance,
                       accounted_yaw_rad=accounted,
                       wrapped_balance_error_rad=wrap(yaw-t['start_yaw_rad']-accounted)))

    def finish(self, now, yaw, variance, reason):
        if self.turn is not None:
            self.snapshot(now, yaw, variance, 'final', reason)
            self.turn = None

    def command(self, now, command, yaw, variance, reason='command'):
        direction = 0 if abs(command) <= 1e-9 else (1 if command > 0 else -1)
        t = self.turn
        if t is not None and t['stop_sec'] is None and direction != t['direction']:
            t['stop_sec'] = now
            self.snapshot(now, yaw, variance, 'stop', reason)
        if direction and (t is None or t['stop_sec'] is not None):
            self.finish(now, yaw, variance, 'next_turn')
            self.counter += 1
            self.turn = dict(
                turn_id=self.counter, start_sec=now, stop_sec=None,
                direction=direction, command_min_abs=abs(command), command_max_abs=abs(command),
                start_yaw_rad=yaw, start_yaw_variance=variance,
                predicted_yaw_rad=0., tag_correction_yaw_rad=0.,
                post_stop_tag_correction_yaw_rad=0., accepted_updates=0,
                rejected_updates=0, initialized_tags=0, post_stop_accepted_updates=0,
                prediction_steps=0, prediction_dt_sum=0., prediction_dt_squared_sum=0.,
                angular_process_variance_added=0.,
            )
        elif direction and self.turn is not None:
            self.turn['command_min_abs'] = min(self.turn['command_min_abs'], abs(command))
            self.turn['command_max_abs'] = max(self.turn['command_max_abs'], abs(command))

    def predict(self, now, delta, dt, q_added, yaw, variance):
        t = self.turn
        if t is None:
            return
        if t['stop_sec'] is not None:
            if now - t['stop_sec'] >= self.settle_seconds:
                self.finish(now, yaw, variance, 'settle_window_elapsed')
            return
        t['predicted_yaw_rad'] += delta
        t['prediction_steps'] += 1
        t['prediction_dt_sum'] += dt
        t['prediction_dt_squared_sum'] += dt*dt
        t['angular_process_variance_added'] += q_added

    def update(self, status, before_yaw, after_yaw):
        t = self.turn
        if t is None:
            return
        if status == 'accepted':
            post = t['stop_sec'] is not None
            t['post_stop_accepted_updates' if post else 'accepted_updates'] += 1
            t['post_stop_tag_correction_yaw_rad' if post else 'tag_correction_yaw_rad'] += wrap(after_yaw-before_yaw)
        elif status == 'rejected_gate':
            t['rejected_updates'] += 1
        elif status == 'initialized':
            t['initialized_tags'] += 1
