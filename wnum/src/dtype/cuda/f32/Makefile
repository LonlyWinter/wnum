

.DELETE_ON_ERROR:

DIR ?= .

${DIR}/libf32.so: utils.o f32.o
	nvcc -shared -Xcompiler -fPIC -arch=sm_61 f32.o utils.o -o ${DIR}/libf32.so

utils.o: utils.cu
	nvcc -Xcompiler -fPIC -arch=sm_61 -lcudart -c utils.cu -o utils.o

f32.o: f32.cu
	nvcc -Xcompiler -fPIC -arch=sm_61 -lcudart -c f32.cu -o f32.o


clean:
	rm -rf *.o