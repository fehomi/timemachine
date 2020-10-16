import argparse

import numpy as np
import time
import os

import logging
import pickle
from concurrent import futures

import grpc

import service_pb2
import service_pb2_grpc

from threading import Lock

from timemachine.lib import custom_ops

from simtk.openmm import app # debug use for model writing

class Worker(service_pb2_grpc.WorkerServicer):

    def Simulate(self, request, context):

        if request.precision == 'single':
            precision = np.float32
        elif request.precision == 'double':
            precision = np.float64
        else:
            raise Exception("Unknown precision")

        simulation = pickle.loads(request.simulation)

        bps = []
        pots = []

        for potential in simulation.potentials:
            bps.append(potential.bound_impl()) # get the bound implementation

        # reseed if the seed is zero.
        if simulation.integrator.seed == 0:
            simulation.integrator.seed = np.random.randint(0, np.iinfo(np.int32).max)

        intg = simulation.integrator.impl()

        ctxt = custom_ops.Context(
            simulation.x,
            simulation.v,
            simulation.box,
            intg,
            bps
        )

        lamb = request.lamb

        minimize_schedule = np.concatenate([
            np.linspace(0.5, lamb, 500),                 # insertion
            np.linspace(lamb, lamb, request.prep_steps)  # equilibration
        ])

        for step, minimize_lamb in enumerate(minimize_schedule):
            ctxt.step(minimize_lamb)

        _, du_dl, _ = bps[-2].execute(ctxt.get_x_t(), simulation.box, lamb)

        if abs(du_dl) > 5000:
            with open("bad_debug_minimize_"+str(simulation.integrator.seed)+".pdb", "w") as out_file:
                print("bad minimize du_dl found for seed", simulation.integrator.seed)
                model = app.PDBFile("holy_debug.pdb")
                app.PDBFile.writeHeader(model.topology, out_file)
                app.PDBFile.writeModel(model.topology, ctxt.get_x_t()*10, out_file, step)
                app.PDBFile.writeFooter(model.topology, out_file)

        energies = []
        frames = []

        if request.observe_du_dl_freq > 0:
            du_dl_obs = custom_ops.AvgPartialUPartialLambda(bps, request.observe_du_dl_freq)
            ctxt.add_observable(du_dl_obs)

        if request.observe_du_dp_freq > 0:
            du_dps = []
            # for name, bp in zip(names, bps):
            # if name == 'LennardJones' or name == 'Electrostatics':
            for bp in bps:
                du_dp_obs = custom_ops.AvgPartialUPartialParam(bp, request.observe_du_dp_freq)
                ctxt.add_observable(du_dp_obs)
                du_dps.append(du_dp_obs)

        # dynamics
        # model = app.PDBFile("holy_debug.pdb")
        # with open("debug_dynamics_"+str(simulation.integrator.seed)+".pdb", "w") as out_file:
            # app.PDBFile.writeHeader(model.topology, out_file)

        for step in range(request.prod_steps):
            if step % 100 == 0:
                u = ctxt.get_u_t()
                energies.append(u)

            if request.n_frames > 0:
                interval = max(1, request.prod_steps//request.n_frames)
                if step % interval == 0:
                    frames.append(ctxt.get_x_t())

            ctxt.step(lamb)

        # app.PDBFile.writeFooter(model.topology, out_file)


        # if step % 500 == 0:
        # _, du_dl, _ = bps[-1].execute(ctxt.get_x_t(), simulation.box, lamb)
        # if abs(du_dl) > 5000:
            # print("bad final du_dl found for seed", simulation.integrator.seed)
            # app.PDBFile.writeModel(model.topology, ctxt.get_x_t()*10, out_file, step)


        frames = np.array(frames)

        if request.observe_du_dl_freq > 0:
            avg_du_dls = du_dl_obs.avg_du_dl()
        else:
            avg_du_dls = None

        # if abs(avg_du_dls) > 5000:
            # print("bad avg_du_dl found for seed", simulation.integrator.seed)
            # app.PDBFile.writeModel(model.topology, ctxt.get_x_t()*10, out_file, step)



        if request.observe_du_dp_freq > 0:
            avg_du_dps = []
            for obs in du_dps:
                avg_du_dps.append(obs.avg_du_dp())
        else:
            avg_du_dps = None

        return service_pb2.SimulateReply(
            avg_du_dls=pickle.dumps(avg_du_dls),
            avg_du_dps=pickle.dumps(avg_du_dps),
            energies=pickle.dumps(energies),
            frames=pickle.dumps(frames),
        )


def serve(args):

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    service_pb2_grpc.add_WorkerServicer_to_server(Worker(), server)
    server.add_insecure_port('[::]:'+str(args.port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Worker Server')
    parser.add_argument('--gpu_idx', type=int, required=True, help='Location of all output files')
    parser.add_argument('--port', type=int, required=True, help='Either single or double precision. Double is 8x slower.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)

    logging.basicConfig()
    serve(args)
