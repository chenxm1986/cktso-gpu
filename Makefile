all:
	g++ -O3 demo.cpp -L. -lcktsogpu -lcktso -o demo
	g++ -O3 demo_l.cpp -L. -lcktsogpu_l -lcktso_l -o demo_l
	g++ -O3 demo_c.cpp -L. -lcktsogpu -lcktso -o demo_c
	g++ -O3 demo_lc.cpp -L. -lcktsogpu_l -lcktso_l -o demo_lc