#include <stdio.h>
#include "etat.h"

using namespace std;

int main(int argv, char* argc[]){
    Etat etat = Etat();
    cout << etat.actualiser() << endl;
    return 0;
}