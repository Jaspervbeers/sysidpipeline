'''
Script to parse string equations of symbols associated with the columns of a Pandas DataFrame

Written by Jasper van Beers
'''
# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import numpy as np
import re
from io import StringIO
import tokenize


def InputParser(x, Data):
    '''Recursive function to resolve symbolic string equation for a Pandas DataFrame 

    :param x: Symbolic string equation to resolve
    :param Data: Pandas DataFrame whose columns correspond to the symbolic variables. The rows denote observations of the corresponding variables. 
    :return: Array of solved equation for each of the rows of param Data
    '''
    # Function to create a generator of all occurrences of pattern, 
    # p in string s. 
    def findall(p, s):
        i = s.find(p)
        while i != -1:
            yield i
            i = s.find(p, i+1)


    # Replace '-' with + and '+' with -
    # This is necessary due to the nature
    # in which the equation is evaluated
    # The sums of the RHS are taken before
    # the LHS, so if there is a subtraction, the
    # signs in the RHS need to be adjusted such that
    # they are in agreement with the original equation
    def replaceMinus(string, stringRef):
        minusIdx = list(findall('-', stringRef))
        plusIdx = list(findall('+', stringRef))
        for i in minusIdx:
            string = string[:i] + '+' + string[i+1:]
        for i in plusIdx:
            string = string[:i] + '-' + string[i+1:]
        return string


    # Function to add vectors Data[a] and Data[b] elementwise
    def add(a, b, Data, *args):
        return np.add(InputParser(a, Data), InputParser(b, Data))


    # Function to subtract vectors Data[a] and Data[b] elementwise 
    def subtract(a, b, Data, *args):
        if a == '':
            a = float(0)
        return np.subtract(InputParser(a, Data), InputParser(b, Data))


    # Function to multiply vectors Data[a] and Data[b] elementwise
    def multiply(a, b, Data, *args):
        return np.multiply(InputParser(a, Data), InputParser(b, Data))
    

    # Function to divide vectors Data[a] and Data[b] elementwise
    def divide(a, b, Data, *args):
        den = InputParser(b, Data)
        if den.all() and ~np.isnan(den.any()):
            return np.divide(InputParser(a, Data), InputParser(b, Data))
        else:
            return np.nan


    # Function to raise the elements of vector Data[a] to the corresponding
    # elements in Data[b] (i.e. elementwise powers)
    def power(a, b, Data, *args):
        try:
            out = np.power(float(InputParser(a, Data)), InputParser(b, Data))
        except TypeError:
            out = np.power(InputParser(a, Data), InputParser(b, Data))
        return out


    # Function to find outermost brackets in string x
    def findBr(x):
        # Convert string in a list of tokens which can be searched 
        lst = [token[1] for token in tokenize.generate_tokens(StringIO(x).readline) if token[1]]
        # Keep track of their index in the string, since some tokens (e.g. 15) are represented
        # by a single index in lst but are represented by multiple indices in x
        lenLst = [len(l) for l in lst]
        oC = 1
        cC = 0
        # Extract the first occurrence of an opening bracket. 
        startIdx = np.where(np.array(lst) == '(')[0][0]
        endIdx = None
        # Loop through the string from the point of the opening bracket, keeping track of
        # how many '(' and ')' we have encountered. Once number of '(' = number of ')' we
        # have found the pair of brackets. 
        for i in np.arange(startIdx + 1, len(lst)):
            if lst[i] == '(':
                oC += 1
            elif lst[i] == ')':
                cC += 1
            if oC == cC:
                endIdx = i
                break
        # Convert the indices to be reflective of the positions in the string x
        startIdx = sum(lenLst[0:startIdx])
        endIdx = sum(lenLst[0:endIdx])
        return startIdx, endIdx


    # Function to remove the operators within brackets from being checked
    def removeBr(s):
        if '(' in s:
            srtIdx, endIdx = findBr(s)
            # Replace brackets with dummy variable '~' so that internal
            # operators will not be checked 
            s = s[0:srtIdx] + '~'*(endIdx - srtIdx + 1) + s[endIdx+1:]
            s = removeBr(s)
        return s


    # Function to parse string x 
    def parse(x, xref, Data):
        operations = {'+':add, '-':subtract, '*':multiply, '/':divide, '^':power}

        # Check operations
        if '+' in xref or '-' in xref:
            opSplit = re.compile(r'[\+\-]').split(xref)
        elif '*' in xref or '/' in xref:
            opSplit = re.compile(r'[\*\/]').split(xref)
        elif '^' in xref:
            opSplit = re.compile(r'[\^]').split(xref)
        else:
            try:
                out = Data[x]
                return out
            except KeyError:
                if x.replace('.', '').isdigit():
                    return float(x)
                else:
                    if x == '':
                        return 1
                    else:
                        raise KeyError('Parsed string {} does not match any keys in inputted data frame'.format(x))

        idx = len(opSplit[0])
        op = x[idx]
        LHS = x[:idx]
        RHS = x[idx+1:]
        if op == '-':
            RHS = replaceMinus(RHS, removeBr(RHS))

        out = operations[op](LHS, RHS, Data)

        return out


    # function to check nested brackets    
    def checkBr(x):
        xref = x
        if '(' in x:
            s, e = findBr(x)
            # Check if brackets are first and last element
            if s == 0 and e == len(x) - 1:
                x = x[s+1:e]
                # If so, strip brackets and parse string inside
                x, xref = checkBr(x)
            else:
                # If not, then parse the whole equation while ignoring
                # operators inside the brackets
                xref = removeBr(x)
        return x, xref


    # Strip whitespace if string
    try:
        x = x.replace(' ', '')
        # Check if x is a key
        try:
            out = Data[x]
        except KeyError:
            # Check if x is a constant
            if x.replace('.', '').isdigit():
                return float(x)

            # Check brackets
            x, xref = checkBr(x)
            out = parse(x, xref, Data)

    # If x is not a string, then it is a value/array of values
    except AttributeError:
        return x

    return out