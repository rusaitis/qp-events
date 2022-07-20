def Tic():
    ''' Homemade version of matlab tic and toc functions '''
    import time

    if 'startTime_for_tictoc' in globals():
        elapsedtime = time.time() - startTime_for_tictoc
        print("Elapsed time is " + str(round(elapsedtime,3)) + " seconds.")
        return round(elapsedtime,3)
    else:
        # print "Toc: start time not set"
        # global startTime_for_tictoc
        startTime_for_tictoc = time.time()
