All:
	g++ -O3 demo.cpp -L. -lcktsogpu -lcktso -o demo
	g++ -O3 demo_l.cpp -L. -lcktsogpu_l -lcktso_l -o demo_l