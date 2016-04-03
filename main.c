//
//  main.c
//  Projet
//
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "openCLWrapper.h"

#include "const.h"
#include "exercice1.h"
#include "exercice2.h"
#include "exercice3.h"
#include "exercice4.h"
#include "exercice5.h"
#include "exercice6.h"


typedef enum {
    MAX_IN_ARRAY = 1,
    MIN_MAX_IN_ARRAY,
    POS_IN_ARRAY,
    POS_IN_MATRIX,
    M_POS_IN_MATRIX,
    MOST_FREQUENT_IN_MATRIX,
} exercice;

// utilisation
// ./projetgpgpu ex
int main( int argc, const char * argv[] ) {

    int debug = 0;
    
    srand( 42 );
    
    int exo, type, N, val, width, height;
    
    if ( debug ) {
        exo = 2;
        type = 3;
        N = pow(2, 10);
        findMinMaxValueInArray( type, N );
        return EXIT_SUCCESS;
    } else {
        if ( argc <= 2) {
            printf( "Utilisation :\n" );
            printf( "./projetgpgpu numEx type [args...]\n" );
            printf( "\tnumEx: numéro de l'exercice\n" );
            printf( "\t\t1: Maximum dans un tableau\n" );
            printf( "\t\t2: Minimum et maximum dans un tableau\n" );
            printf( "\t\t3: Position d'un entier dans un tableau\n" );
            printf( "\t\t4: Position d'un entier dans une matrice\n" );
            printf( "\t\t5: Position de M entiers dans une matrice\n" );
            printf( "\t\t6: L'entier le plus fréquent dans une matrice\n" );
            printf( "\ttype: type d'algorithme\n" );
            printf( "\t\t1: sequentiel\n" );
            printf( "\t\t2: GPU\n" );
            printf( "\t\t3: GPU optimisé\n" );
            printf( "\t[args...]: dépend de l'exercice, appeler l'exercice sans argument pour obtenir plus d'info\n" );
            printf( "\t\tex: ./projetgpgpu 1 pour des infos sur l'ex 1...\n" );
            
            return EXIT_FAILURE;
        }
        
        exo = atoi( argv[1] );
        
        if ( exo < 1 || exo > 6 ) {
            printf( "Cet exercice n'éxiste pas (choisir entre 1 et 6)\n");
            return EXIT_FAILURE;
        }

        type = atoi( argv[2] );
        if ( type < 1 || type > 3 ) {
            printf( "Ce type d'algo n'éxiste pas (choisir entre 1 et 3)\n");
            return EXIT_FAILURE;
        }
    }
    
    switch ( exo ) {
        case MAX_IN_ARRAY:
            if ( argc < 4 || argc > 4 ) {
                printf("usage: ./projetgpu 1 type N\n");
                printf("\tN: taille du tableau\n");
                return EXIT_FAILURE;
            }
        
            N = atoi( argv[3] );
            findMaxValueInArray( type, N );
        break;
        
        case MIN_MAX_IN_ARRAY:
            if ( argc != 4 ) {
                printf("usage: ./projetgpu 2 type N\n");
                printf("\tN: taille du tableau\n");
                return EXIT_FAILURE;
            }
            
            N = atoi( argv[3] );
            findMinMaxValueInArray( type, N );
            break;
            
        case POS_IN_ARRAY:
            if ( argc != 5 ) {
                printf("usage: ./projetgpu 3 type N val\n");
                printf("\tN: taille du tableau\n");
                printf("\tval: entier à retrouver dans le tableau\n");
                return EXIT_FAILURE;
            }
            
            N = atoi( argv[3] );
            val = atoi( argv[4] );
            findValueInArray( type, N, val );
            break;
            
        case POS_IN_MATRIX:
            if ( argc != 6 ) {
                printf("usage: ./projetgpu 4 type width height val\n");
                printf("\twidth: largeur de la matrice\n");
                printf("\theight: hauteur de la matrice\n");
                printf("\tval: entier à retrouver dans le tableau\n");
                return EXIT_FAILURE;
            }
        
            width = atoi( argv[3] );
            height = atoi( argv[4] );
            val = atoi( argv[5] );
            findValueInMatrix( type, width, height, val );
            break;
            
        case M_POS_IN_MATRIX:
            if ( argc < 6 ) {
                printf("usage: ./projetgpu 5 type width height M...\n");
                printf("\twidth: largeur de la matrice\n");
                printf("\theight: hauteur de la matrice\n");
                printf("\tM: les entiers à retrouver dans le tableau séparées par des espaces\n");
                return EXIT_FAILURE;
            }
            
            width = atoi( argv[3] );
            height = atoi( argv[4] );
            int nbM = argc - 5;
            int* M = ( int* )malloc( sizeof( int ) * nbM );
            for ( int i = 0; i < nbM; i++ ) {
                M[i] = atoi( argv[i + 5] );
            }
            findValuesInMatrix( type, width, height, M, nbM );
            free( M );
            break;
        
        case MOST_FREQUENT_IN_MATRIX:
            if ( argc != 5 ) {
                printf("usage: ./projetgpu 6 type width height\n");
                printf("\twidth: largeur de la matrice\n");
                printf("\theight: hauteur de la matrice\n");
                return EXIT_FAILURE;
            }
            
            width = atoi( argv[3] );
            height = atoi( argv[4] );
            findMostFrequentValueInMatrix( type, width, height );
            break;
        
        default:
            printf( "Nope\n" );
            break;
    }
    
    return EXIT_SUCCESS;
}