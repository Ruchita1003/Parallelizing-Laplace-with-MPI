# Author: Ruchita Nagare

from mpi4py import MPI
import numpy as np

# com world, rank
comm = MPI.COMM_WORLD
my_pe_num = comm.Get_rank()
# flag for breaking out of the while loop
break_flag = 0

# number of rows and columns
ROWS , COLUMNS = 250 , 1000
MAX_TEMP_ERROR = 0.01
temperature      = np.empty(( ROWS+2 , COLUMNS+2 ))
temperature_last = np.empty(( ROWS+2 ,COLUMNS+2  ))

# all PEs will have send and receive arrays
send_array_to_down = np.empty(1002)
recv_array_from_up = np.empty(1002)
send_array_to_up = np.empty(1002)
recv_array_from_down = np.empty(1002)

def initialize_temperature(temp):
    temp[:,:] = 0

    for i in range(1,ROWS+1):
        # index = int(((COLUMNS/comm.Get_size()) * my_pe_num) + i)
        value = ((100/comm.Get_size()) * my_pe_num) + i/10
        temp[i, COLUMNS+1] = value

    if my_pe_num==3:
        for i in range(1,COLUMNS+1):
            temp[ ROWS+1 , i ] = ((100/COLUMNS ) * i)

def output(data):
    data.tofile("plate_full_4_1000.out")

initialize_temperature(temperature_last)

# manager PE (PE0) take iterations input
max_iterations=0
if my_pe_num == 0:
    max_iterations = int(input("Maximum iterations: "))

#broadcast the max_iterations
max_iterations = comm.bcast(max_iterations, root =0)

dt = 100
iteration = 1

print("Starting while loop", flush = True)
# all PEs exchange arrays and compute temperature as long as iteration < max_iterations 
while ( iteration < max_iterations ):
    print("Starting exchange and calculations", flush = True)

    # send 250th row of current PE to 0th row of next PE (all send except PE3)
    send_array_to_down = temperature_last[-2,:]
    if my_pe_num!=3:
        comm.Send(send_array_to_down, dest = my_pe_num+1)

    # send 1st row of current PE to 251st row of previous PE (all send except PE0)
    send_array_to_up = temperature_last[1,:]
    if my_pe_num!=0:
        comm.Send(send_array_to_up, dest = my_pe_num-1)

    # receive 250th row of previous PE into the 0th row of current PE (all receive except PE0)
    if my_pe_num!=0:
        comm.Recv(recv_array_from_up, source = my_pe_num-1, tag=MPI.ANY_TAG)
        # put recv_array_from_up into 0th of temperature array
        temperature_last[0] = recv_array_from_up

    # receive 1st row of next PE into the 251st row of current PE (all receive except PE3)
    if my_pe_num!=3:
        comm.Recv(recv_array_from_down, source = my_pe_num+1, tag=MPI.ANY_TAG)
        # put recv_array_from_down into 251st temperarture array
        temperature_last[-1] = recv_array_from_down

    # compute temperature from temperature_last
    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            temperature[ i , j ] = 0.25 * ( temperature_last[i+1,j] + temperature_last[i-1,j] +
            temperature_last[i,j+1] + temperature_last[i,j-1])

    # calculate dt
    dt = 0
    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            dt = max( dt, temperature[i,j] - temperature_last[i,j])
            temperature_last[ i , j ] = temperature [ i , j ]

    # A list that hold dt of all PEs
    dt_list = []
    if my_pe_num!=0:
        # PE1,PE2,PE3 will send their dt to PE0
        comm.send(dt, dest = 0)

    else:
        # PE0 appends its dt in 0th of dt_list
        dt_list.append(dt)
        # PE 0 receives dt from all other PEs and appends it in dt_list
        for pe in range(1,comm.Get_size()):
            recv_dt = 100
            recv_dt = comm.recv(source = pe, tag=MPI.ANY_TAG)
            dt_list.append(recv_dt)
        print("dt_list: " + str(dt_list), flush = True)
        # PE checks max of dt_list < 0.01
        if max(dt_list)<MAX_TEMP_ERROR:
            # if yes, then change break_flag to 1
            break_flag = 1
        # PE0 sends the break flag to all other PEs
        for pe in range(1,comm.Get_size()):
            comm.send(break_flag, dest = pe)

    # all PEs need to be synchronozed here, else they can receive a stale flag
    comm.barrier()
    if my_pe_num!=0:
        # PE1,PE2,PE3 receive the break_flag from PE0
        break_flag = comm.recv(source = 0, tag=MPI.ANY_TAG)
    if break_flag==1:
        # if break_flag is 1, the while loop is breaked (i.e. convergence)
        break

    print("PE " + str(my_pe_num) + " completete iteration: " + str(iteration))
    iteration += 1

if my_pe_num==0:
    # deleting the ghost cells
    temperature_last=np.delete(temperature_last, (251), axis=0)
    temperature_last=np.delete(temperature_last, (0), axis=0)
    # data_gath_array will hold the entire final array from all PEs
    # first index of data_gath_array will contain PE0's temperature_last
    data_gath_array = temperature_last

for pe in range(1,comm.Get_size()):
    comm.barrier()
    # All PE1,PE2,PE3 will send their temperature_last array sequentially to PE0
    if my_pe_num==pe:
        temperature_last=np.delete(temperature_last, (251), axis=0)
        temperature_last=np.delete(temperature_last, (0), axis=0)
        comm.Send(temperature_last, dest = 0)
    # PE0 will receive temperature_last from all other PEs and concatenate with data_gath_array
    if my_pe_num==0:
        received_temperature = np.empty((250,1002))
        comm.Recv(received_temperature, source = pe, tag=MPI.ANY_TAG)
        data_gath_array = np.concatenate((data_gath_array, received_temperature), axis=0)
        
print("reached here", flush =True)
# PE0 will call the output function with data_gath_array as argument 
if my_pe_num==0:
    print(data_gath_array.shape, flush = True)
    print(data_gath_array.shape, flush = True)
    output(data_gath_array)