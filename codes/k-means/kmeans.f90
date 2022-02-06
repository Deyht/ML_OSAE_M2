!############################### ################################
! K-means exercise for the M2-OSAE Machine Learning lessons
! contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
!############################### ################################


module utils
	implicit none

contains
	! This Function return a distance in ndim dimension
	! between two points (arguments are table with all the dimensions) 
	function dist(dat, cent, ndim)
		real :: dist
		real, intent(in) :: dat(:), cent(:)
		integer, intent(in) :: ndim
		integer :: i

		dist = 0.0
		do i = 1, ndim
			dist = dist + (dat(i)-cent(i))**2
		end do
	end function dist

end module utils



program kmeans

	use utils
	implicit none

	! Usefull data, feel free to add the ones you may need
	! for your own implementation of the algorithm
	real, allocatable :: input_data(:,:), centers(:,:)
	integer, allocatable :: nb_points_per_center(:)
	integer :: nb_dim, nb_data
	real :: rand
	integer :: nb_k
	integer :: i, j

	nb_k = 4

	! This entry file must be edited to change to other
	! number of dimension. The code must be re-compiled !
	open(unit = 10, file="kmeans_input_file_2d.dat")

	! Read the dimensions of the data in the file
	read(10, *) nb_dim, nb_data
	allocate(input_data(nb_data, nb_dim))

	write(*,*) nb_dim, nb_data

	! Load all the data
	do i = 1, nb_dim
		read(10, *) input_data(:, i)
	end do

	close(10)

	! Allocate the tables according to the dimension
	! gave in the input file
	allocate(centers(nb_k, nb_dim))
	allocate(nb_points_per_center(nb_k))

	! The origin of the centers are selected randomly
	! to the position of some points in the dataset
	do i=1, nb_k
		call random_number(rand)
		centers(i, :) = input_data(int(rand*nb_data),:)
	end do

	!############################### ################################
	!     Main loop, until the new centers do not move anymore
	!############################### ################################


		!############################### ################################
		!         Association phase, loop on the data points
		!############################### ################################
		



		!############################### ################################
		!           Update phase, calculate the new centers
		!############################### ################################
		



	
	!############################### ################################
	!      Save the ending centroid position for visualization
	!############################### ################################	
	open(unit = 10, file="kmeans_output_2d.dat")
	
	write(10,*) nb_dim, nb_k
	
	do i = 1, nb_k
		write (10,*) centers(i, :)
	end do
	
	deallocate(input_data)
	deallocate(centers)
	deallocate(nb_points_per_center)
	
	

end program kmeans








