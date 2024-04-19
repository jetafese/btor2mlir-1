import sys


class CexWitnessGenerator(object):

  def __init__(self, name='CexWitnessGenerator', help='Generate btor witness from SeaHorn cex'):
    self.name = name
    self.help = help
    self.states = list()
    self.inputs = dict()

  def get_value(self, value, width):
    return format(value, '0{}b'.format(width))

  def mk_arg_parser(self, ap):
    ap.add_argument('-o',
                    dest='out_file',
                    metavar='FILE',
                    help='Output file name',
                    default=None)
    ap.add_argument('in_file',
                    metavar='FILE',
                    help='Input file')

    return ap

  def run(self, args=None):
    # default output destination is file `cex.txt`
    if args.out_file is None:
        args.out_file = 'cex.txt'
    print('Creating', args.out_file, '...')
    # keeps track of seen states and inputs
    seenStates = dict()
    seenInputs = dict()
    seenArrays = dict()
    # keeps track of states and inputs per frame
    states = list()
    inputs = list()
    arrays = list()
    # register violated assertion
    violatedProperty = 0
    # read each line
    with open(args.in_file, errors='replace') as input:
      for line in input:
        if line.__contains__('[sea]'):
          if not line.__contains__('__VERIFIER_assert'):
            continue
          splitSeaLine = line.split(':')
          violatedProperty = int(splitSeaLine[1].split(',')[0].split()[0])
          continue
        splitLine = line.split(',')
        name = splitLine[0].split()[0]
        btorId = int(splitLine[1].split()[0])
        btorValue = int(splitLine[2].split()[0])
        bvWidth = int(splitLine[3].split()[0])
        # print(f'{name}, {btorId}, {btorValue}, {bvWidth}, {self.get_value(btorValue, bvWidth)}')
        if name == 'array':
          btorIndex = int(splitLine[2].split()[0])
          btorValue = int(splitLine[3].split()[0])
          bvWidth = int(splitLine[4].split()[0])
          # print(f'{name}, {btorId}, {btorIndex}, {btorValue}, {bvWidth}, {self.get_value(btorValue, bvWidth)}')
          idx_val = tuple((self.get_value(btorIndex, bvWidth), self.get_value(btorValue, bvWidth)))
          if btorId in seenArrays:
            # print('repeat state: ', btorId)
            prev = seenArrays[btorId]
            seenArrays[btorId] = prev + list(idx_val)
          else:
            # print('got state: ', btorId)
            seenArrays[btorId] = list(idx_val)
        elif name == 'state':
          if btorId in seenStates:
            # print('repeat state: ', btorId)
            states.append(seenStates)
            arrays.append(seenArrays)
            inputs.append(seenInputs)
            seenStates = dict(); seenInputs = dict(); seenArrays = dict()
            seenStates[btorId] = self.get_value(btorValue, bvWidth)
          else:
            # print('got state: ', btorId)
            seenStates[btorId] = self.get_value(btorValue, bvWidth)
        else:
          if btorId in seenInputs:
            # print('repeat input: ', btorId)
            states.append(seenStates)
            arrays.append(seenArrays)
            inputs.append(seenInputs)
            seenStates = dict(); seenInputs = dict(); seenArrays = dict()
            seenInputs[btorId] = self.get_value(btorValue, bvWidth)
          else:
            # print('got input: ', btorId)
            seenInputs[btorId] = self.get_value(btorValue, bvWidth)

    # write to output file
    f = open(args.out_file, 'w')
    frame = 0
    # print header
    f.write(f'sat\n')
    f.write(f'b{violatedProperty}\n') # which property is violated? (b0, b1, j0,...)

    if seenStates or seenInputs or seenArrays:
      states.append(seenStates)
      arrays.append(seenArrays)
      inputs.append(seenInputs)
    # print(inputs)
    # print(states)
      for (s, a, i) in zip(states, arrays, inputs):
        # print(s, i)
        f.write(f'#{frame}\n')
        if s:
          for k, v in s.items():
            f.write(f'{k} {v}\n')
        if a:
          for k, v in a.items():
            for c in range(len(v)):
              if c % 2 == 0:
                f.write(f'{k} [{v[c]}] {v[c+1]}\n')
        f.write(f'@{frame}\n')
        if i:
          print(i)
          for k, v in i.items():
            f.write(f'{k} {v}\n')
        frame += 1
    else:
      for i in range(21):
        f.write(f'#{i}\n')
        f.write(f'@{i}\n')

    f.write('.\n')
    f.close()
    return 0

  def main(self, argv):
    import argparse

    ap = argparse.ArgumentParser(prog=self.name, description=self.help)
    ap = self.mk_arg_parser(ap)

    args = ap.parse_args(argv)
    return self.run(args)

def main():
  cmd = CexWitnessGenerator()
  return cmd.main(sys.argv[1:])


if __name__ == '__main__':
  sys.exit(main())