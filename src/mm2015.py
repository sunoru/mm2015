# coding=utf-8
import sys
from main_algorithm import main
from vehicle import cities

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Invalid Input"
        sys.exit()
    if sys.argv[1] in cities:
        main(sys.argv[1])
