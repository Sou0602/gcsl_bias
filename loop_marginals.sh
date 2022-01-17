for s in 0 
do 
    for e in pusher door lunar pointmass_rooms pointmass_empty
    do
	echo Environment $e Seed $s
        #python experiments/gcsl_example.py -S $s -E $e &
	#python experiments/gcsl_example_n11.py -S $s -E $e &
	python experiments/gcsl_example_ne.py -S $s -E $e &
    done
    wait
done

#
#for s in 1
#do 
#    for e in pointmass_empty pointmass_rooms lunar pusher door 
#    do
#	echo Environment $e Seed $s
#        python experiments/gcsl_example_n.py -S $s -E $e &
#    done
#done
#
#for s in 2
#do 
#    for e in pointmass_empty pointmass_rooms lunar pusher door 
#    do
#	echo Environment $e Seed $s
#        python experiments/gcsl_example_n.py -S $s -E $e &
#    done
#done

#for k in 0 1 2 3 4 5
#do
#	python experiments/gcsl_example.py -K $k &
#	
#done
#wait

#for k in 0 1 2 3 4 5
#do
#	python experiments/gcsl_example_n11.py -K $k &
#	
#done
#wait



exit 0
