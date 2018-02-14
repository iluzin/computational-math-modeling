# -*- coding: utf-8 -*-

import copy
import numpy as np

class Verlet(object):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            exec 'self.{} = value'.format(key)
    
    def __call__(self, bodies):
        G = 6.67408e-11
        acceleration = np.zeros((len(bodies), 2))
        for i in xrange(len(bodies)):
            for j in xrange(len(bodies)):
                if i != j:
                    shift = np.asarray(bodies[j].pos()) - np.asarray(bodies[i].pos())
                    acceleration[i, :] += G * bodies[j].mass * shift / np.linalg.norm(shift) ** 3
        out = []
        for i in xrange(len(self.tspan)):
            out.append(copy.deepcopy(bodies))
        out = np.asarray(out)
        for i in xrange(1, out.shape[0]):
            tau = self.tspan[i] - self.tspan[i - 1]
            for j in xrange(len(bodies)):
                out[i, j].x = out[i - 1, j].x + out[i - 1, j].dxdt * tau + 0.5 * acceleration[j, 0] * tau ** 2
                out[i, j].y = out[i - 1, j].y + out[i - 1, j].dydt * tau + 0.5 * acceleration[j, 1] * tau ** 2
            for j in xrange(len(bodies)):
                acceleration_next = np.zeros(2)
                for k in xrange(len(bodies)):
                    if j != k:
                        shift = np.asarray((out[i, k].x, out[i, k].y)) - np.asarray((out[i, j].x, out[i, j].y))
                        acceleration_next += G * out[i, k].mass * shift / np.linalg.norm(shift) ** 3
                out[i, j].dxdt = out[i - 1, j].dxdt + 0.5 * (acceleration_next[0] + acceleration[j, 0]) * tau
                out[i, j].dydt = out[i - 1, j].dydt + 0.5 * (acceleration_next[1] + acceleration[j, 1]) * tau
                acceleration[j, :] = acceleration_next
        return out

class VerletThreading(object):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            exec 'self.{} = value'.format(key)
    
    def __call__(self, bodies):
        G = 6.67408e-11
        times = np.arange(n) * delta_t
        numberOfPlanets = ip.size
        out = np.empty((times.size, numberOfPlanets * 2))
        out[0] = np.concatenate((ip, iv))
        dt=delta_t
        dimension = len(ip) // len(m)

        def solveForOneBody(body, event, controlevent, loopthread):
            cur_acceleration = acceleration(body, m, out[0, :numberOfPlanets])
            for j in np.arange(n - 1) + 1:
                event.set()
                loopthread.set()
                controlevent.wait()
                controlevent.clear()
                out[j, body*dimension:(body+1)*dimension] = \
                    (out[j-1, body*dimension:(body+1)*dimension]
                     + out[j-1, numberOfPlanets+body*dimension:numberOfPlanets+(body+1)*dimension] * dt
                     + 0.5 * cur_acceleration * dt**2)

                event.set()
                loopthread.set()
                controlevent.wait()
                controlevent.clear()

                next_acceleration = acceleration(body, m, out[j, :numberOfPlanets])
                out[j, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension] = \
                    (out[j-1, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension]
                     + 0.5 * (cur_acceleration + next_acceleration) * dt)
                cur_acceleration = next_acceleration
                event.set()
                loopthread.set()
                controlevent.wait()
                controlevent.clear()
            return

        events = []
        for body in m:
            events.append(threading.Event())
            events[-1].clear()
        events[0].set()

        isfreeevent = threading.Event()
        controlevent = threading.Event()
        loopevent = threading.Event()
        controlthread = threading.Thread(target=Sync, args=(events, controlevent, loopevent, isfreeevent))
        controlthread.start()

        t=[]
        for i in range(m.size):
            t.append(threading.Thread(target=solveForOneBody, args=(i,events[i], controlevent, loopevent)))
            t[i].start()
        for i in range(m.size):
            t[i].join()
        loopevent.set()
        isfreeevent.set()
        controlthread.join()

        # print(out)
        return out

class VerletMultiprocessing(object):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            exec 'self.{} = value'.format(key)
    
    def __call__(self, bodies):
        G = 6.67408e-11
        acceleration = np.zeros((len(bodies), 2))
        for i in xrange(len(bodies)):
            for j in xrange(len(bodies)):
                if i != j:
                    shift = np.asarray(bodies[j].pos()) - np.asarray(bodies[i].pos())
                    acceleration[i, :] += G * bodies[j].mass * shift / np.linalg.norm(shift) ** 3
        out = []
        for i in xrange(len(self.tspan)):
            out.append(copy.deepcopy(bodies))
        out = np.asarray(out)
        for i in xrange(1, out.shape[0]):
            tau = self.tspan[i] - self.tspan[i - 1]
            for j in xrange(len(bodies)):
                out[i, j].x = out[i - 1, j].x + out[i - 1, j].dxdt * tau + 0.5 * acceleration[j, 0] * tau ** 2
                out[i, j].y = out[i - 1, j].y + out[i - 1, j].dydt * tau + 0.5 * acceleration[j, 1] * tau ** 2
            for j in xrange(len(bodies)):
                acceleration_next = np.zeros(2)
                for k in xrange(len(bodies)):
                    if j != k:
                        shift = np.asarray((out[i, k].x, out[i, k].y)) - np.asarray((out[i, j].x, out[i, j].y))
                        acceleration_next += G * out[i, k].mass * shift / np.linalg.norm(shift) ** 3
                out[i, j].dxdt = out[i - 1, j].dxdt + 0.5 * (acceleration_next[0] + acceleration[j, 0]) * tau
                out[i, j].dydt = out[i - 1, j].dydt + 0.5 * (acceleration_next[1] + acceleration[j, 1]) * tau
                acceleration[j, :] = acceleration_next
        return out

class VerletCython(object):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            exec 'self.{} = value'.format(key)
    
    def __call__(self, bodies):
        G = 6.67408e-11
        acceleration = np.zeros((len(bodies), 2))
        for i in xrange(len(bodies)):
            for j in xrange(len(bodies)):
                if i != j:
                    shift = np.asarray(bodies[j].pos()) - np.asarray(bodies[i].pos())
                    acceleration[i, :] += G * bodies[j].mass * shift / np.linalg.norm(shift) ** 3
        out = []
        for i in xrange(len(self.tspan)):
            out.append(copy.deepcopy(bodies))
        out = np.asarray(out)
        for i in xrange(1, out.shape[0]):
            tau = self.tspan[i] - self.tspan[i - 1]
            for j in xrange(len(bodies)):
                out[i, j].x = out[i - 1, j].x + out[i - 1, j].dxdt * tau + 0.5 * acceleration[j, 0] * tau ** 2
                out[i, j].y = out[i - 1, j].y + out[i - 1, j].dydt * tau + 0.5 * acceleration[j, 1] * tau ** 2
            for j in xrange(len(bodies)):
                acceleration_next = np.zeros(2)
                for k in xrange(len(bodies)):
                    if j != k:
                        shift = np.asarray((out[i, k].x, out[i, k].y)) - np.asarray((out[i, j].x, out[i, j].y))
                        acceleration_next += G * out[i, k].mass * shift / np.linalg.norm(shift) ** 3
                out[i, j].dxdt = out[i - 1, j].dxdt + 0.5 * (acceleration_next[0] + acceleration[j, 0]) * tau
                out[i, j].dydt = out[i - 1, j].dydt + 0.5 * (acceleration_next[1] + acceleration[j, 1]) * tau
                acceleration[j, :] = acceleration_next
        return out

class VerletOpenComputingLanguage(object):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.iteritems():
            exec 'self.{} = value'.format(key)
    
    def __call__(self, bodies):
        G = 6.67408e-11
        acceleration = np.zeros((len(bodies), 2))
        for i in xrange(len(bodies)):
            for j in xrange(len(bodies)):
                if i != j:
                    shift = np.asarray(bodies[j].pos()) - np.asarray(bodies[i].pos())
                    acceleration[i, :] += G * bodies[j].mass * shift / np.linalg.norm(shift) ** 3
        out = []
        for i in xrange(len(self.tspan)):
            out.append(copy.deepcopy(bodies))
        out = np.asarray(out)
        for i in xrange(1, out.shape[0]):
            tau = self.tspan[i] - self.tspan[i - 1]
            for j in xrange(len(bodies)):
                out[i, j].x = out[i - 1, j].x + out[i - 1, j].dxdt * tau + 0.5 * acceleration[j, 0] * tau ** 2
                out[i, j].y = out[i - 1, j].y + out[i - 1, j].dydt * tau + 0.5 * acceleration[j, 1] * tau ** 2
            for j in xrange(len(bodies)):
                acceleration_next = np.zeros(2)
                for k in xrange(len(bodies)):
                    if j != k:
                        shift = np.asarray((out[i, k].x, out[i, k].y)) - np.asarray((out[i, j].x, out[i, j].y))
                        acceleration_next += G * out[i, k].mass * shift / np.linalg.norm(shift) ** 3
                out[i, j].dxdt = out[i - 1, j].dxdt + 0.5 * (acceleration_next[0] + acceleration[j, 0]) * tau
                out[i, j].dydt = out[i - 1, j].dydt + 0.5 * (acceleration_next[1] + acceleration[j, 1]) * tau
                acceleration[j, :] = acceleration_next
        return out
