// Single-issue RV32I core (Zicsr, Zifencei, machine mode only) plus an RVV 1.0 SUBSET (NOT complete RVV).
// Machine-mode CSRs: mstatus(MIE/MPIE, MPP=M), misa, mie, mtvec(direct), mscratch, mepc, mcause, mtval, mip(=0),
// mcountinhibit, mcycle/minstret(+h) and cycle/instret(+h) shadows, mvendorid/marchid/mimpid/mhartid(=0), and read-only
// vl/vtype/vlenb. No U/S modes, PMP, interrupts sources, debug triggers or time CSR; unimplemented CSRs trap (illegal).
// Traps are architectural (mepc/mcause/mtval, jump to mtvec, MIE->MPIE) unless HALT_ON_TRAP=1, a bring-up mode that
// halts on any trap and holds the trap_* pins. IALIGN=32: misaligned control-flow targets trap. Misaligned scalar
// loads/stores are handled in hardware with two aligned word accesses (the spec permits either).
// RVV subset: VLEN=128, ELEN=32, LMUL=1 only, SEW 8/16/32, vstart=0, no masking (vm must be 1),
// tail-undisturbed. Anything else (LMUL!=1, SEW=64 -> vill; masked/other opcodes -> illegal trap).
//   vsetvli/vsetivli/vsetvl; vle{8,16,32}.v / vse{8,16,32}.v (unit stride, EEW==SEW, aligned);
//   OPIVV/OPIVX/OPIVI: vadd vsub(vv,vx) vrsub(vx,vi) vminu/vmin/vmaxu/vmax(vv,vx) vand vor vxor
//   vsll vsrl vsra vmv.v.{v,x,i}; OPMVV/OPMVX: vmul vmacc vredsum.vs vmv.x.s vmv.s.x.
// Vector memory accesses are element-serial on the 32-bit data bus.
module rv32i_core #(
    parameter logic [31:0] RESET_PC = 32'h0000_0000,
    parameter bit HALT_ON_TRAP = 1'b0
) (
    input  logic        clk,
    input  logic        rst_n,

    // Instruction request held valid until ready. Data/fault are sampled on the
    // completion cycle; instruction addresses are 32-bit aligned (IALIGN=32).
    output logic        imem_valid,
    output logic [31:0] imem_addr,
    input  logic        imem_ready,
    input  logic [31:0] imem_rdata,
    input  logic        imem_fault,

    // At most one data transaction is outstanding. Little-endian aligned word
    // bus; byte enables and shifted write data select lanes, reads return word.
    output logic        dmem_valid,
    output logic        dmem_write,
    output logic [31:0] dmem_addr,
    output logic [31:0] dmem_wdata,
    output logic [3:0]  dmem_wstrb,
    input  logic        dmem_ready,
    input  logic [31:0] dmem_rdata,
    input  logic        dmem_fault,

    output logic        halted,
    output logic        trap_valid,
    output logic [31:0] trap_pc,
    output logic [3:0]  trap_cause,
    output logic [31:0] trap_value,

    output logic        retire_valid,
    output logic [31:0] retire_pc,
    output logic [31:0] retire_insn,
    output logic [31:0] retire_next_pc
);
    localparam logic [3:0] CAUSE_INSN_MISALIGNED  = 4'd0;
    localparam logic [3:0] CAUSE_INSN_ACCESS      = 4'd1;
    localparam logic [3:0] CAUSE_ILLEGAL          = 4'd2;
    localparam logic [3:0] CAUSE_BREAKPOINT       = 4'd3;
    localparam logic [3:0] CAUSE_LOAD_MISALIGNED  = 4'd4;
    localparam logic [3:0] CAUSE_LOAD_ACCESS      = 4'd5;
    localparam logic [3:0] CAUSE_STORE_MISALIGNED = 4'd6;
    localparam logic [3:0] CAUSE_STORE_ACCESS     = 4'd7;
    localparam logic [3:0] CAUSE_ENV_CALL         = 4'd11;

    typedef enum logic [3:0] { ST_FETCH, ST_LOAD, ST_LOAD2, ST_STORE, ST_STORE2, ST_HALTED, ST_VLOAD, ST_VSTORE } state_t;
    state_t state;
    logic [31:0] pc;
    logic [31:0] registers [0:31];
    logic [31:0] data_address;
    logic [31:0] store_data, store_data2;
    logic [3:0] store_strobes, store_strobes2;
    logic store_two;
    logic [2:0] load_funct3;
    logic [4:0] load_rd;
    logic [1:0] load_shift;
    logic load_two;
    logic [31:0] load_word0;
    logic [31:0] memory_instruction;
    // RVV subset state.
    logic [127:0] vregs [0:31];
    logic [31:0] vl_reg;
    logic [31:0] vtype_reg;
    logic [5:0]  v_idx;
    logic [4:0]  v_reg;
    // Machine-mode CSR state.
    logic mstatus_mie, mstatus_mpie;
    logic [31:0] mtvec, mepc, mcause, mtval, mscratch, mie_reg, mcountinhibit;
    logic [63:0] mcycle, minstret;
    logic instret_pending;
    wire [1:0]   vsew = vtype_reg[4:3];

    wire [6:0] opcode = imem_rdata[6:0];
    wire [4:0] rd = imem_rdata[11:7];
    wire [2:0] funct3 = imem_rdata[14:12];
    wire [4:0] rs1 = imem_rdata[19:15];
    wire [4:0] rs2 = imem_rdata[24:20];
    wire [6:0] funct7 = imem_rdata[31:25];
    wire [31:0] rs1_value = (rs1 == 0) ? 32'd0 : registers[rs1];
    wire [31:0] rs2_value = (rs2 == 0) ? 32'd0 : registers[rs2];

    wire [31:0] imm_i = {{20{imem_rdata[31]}}, imem_rdata[31:20]};
    wire [31:0] imm_s = {{20{imem_rdata[31]}}, imem_rdata[31:25], imem_rdata[11:7]};
    wire [31:0] imm_b = {{19{imem_rdata[31]}}, imem_rdata[31], imem_rdata[7],
                         imem_rdata[30:25], imem_rdata[11:8], 1'b0};
    wire [31:0] imm_u = {imem_rdata[31:12], 12'b0};
    wire [31:0] imm_j = {{11{imem_rdata[31]}}, imem_rdata[31], imem_rdata[19:12],
                         imem_rdata[20], imem_rdata[30:21], 1'b0};
    wire [31:0] load_effective_address = rs1_value + imm_i;
    wire [31:0] store_effective_address = rs1_value + imm_s;
    wire [31:0] jal_target = pc + imm_j;
    wire [31:0] jalr_target = (rs1_value + imm_i) & 32'hffff_fffe;
    wire [2:0] store_nbytes = (funct3 == 3'b000) ? 3'd1 : (funct3 == 3'b001) ? 3'd2 : 3'd4;
    wire [1:0] store_shift = store_effective_address[1:0];
    wire fetching_second = (state == ST_LOAD2) || (state == ST_STORE2);
    wire [31:0] shifted_load_word = dmem_rdata >> (8 * data_address[1:0]);
    wire [63:0] load_window = fetching_second ? {dmem_rdata, load_word0} : {32'd0, dmem_rdata};
    wire [31:0] load_shifted = load_window >> (8 * load_shift);
    logic [31:0] load_result;
    integer reset_index;

    always_comb begin
        case (load_funct3)
            3'b000: load_result = {{24{load_shifted[7]}}, load_shifted[7:0]};
            3'b001: load_result = {{16{load_shifted[15]}}, load_shifted[15:0]};
            3'b010: load_result = load_shifted[31:0];
            3'b100: load_result = {24'd0, load_shifted[7:0]};
            3'b101: load_result = {16'd0, load_shifted[15:0]};
            default: load_result = 32'd0;
        endcase
    end

    function automatic [31:0] vmask(input [1:0] sew);
        case (sew)
            2'd0: vmask = 32'h0000_00ff;
            2'd1: vmask = 32'h0000_ffff;
            default: vmask = 32'hffff_ffff;
        endcase
    endfunction
    function automatic [3:0] vbytes_mask(input [1:0] sew);
        case (sew)
            2'd0: vbytes_mask = 4'b0001;
            2'd1: vbytes_mask = 4'b0011;
            default: vbytes_mask = 4'b1111;
        endcase
    endfunction
    function automatic [31:0] vsext(input [31:0] x, input [1:0] sew);
        case (sew)
            2'd0: vsext = {{24{x[7]}}, x[7:0]};
            2'd1: vsext = {{16{x[15]}}, x[15:0]};
            default: vsext = x;
        endcase
    endfunction
    function automatic [31:0] velem(input [127:0] v, input [1:0] sew, input integer idx);
        logic [127:0] t;
        begin
            t = v >> (idx * (8 << sew));
            velem = t[31:0] & vmask(sew);
        end
    endfunction
    function automatic [127:0] vput(input [127:0] v, input [1:0] sew, input integer idx, input [31:0] val);
        logic [127:0] m;
        begin
            m = {96'd0, vmask(sew)} << (idx * (8 << sew));
            vput = (v & ~m) | (({96'd0, val & vmask(sew)} << (idx * (8 << sew))) & m);
        end
    endfunction
    function automatic [31:0] valu(input [5:0] f6, input is_m, input [1:0] sew,
                                   input [31:0] a, input [31:0] b, input [31:0] d);
        logic [31:0] sa, sb, sh, r;
        begin
            sa = vsext(a, sew);
            sb = vsext(b, sew);
            sh = b & ((8 << sew) - 1);
            r = 32'd0;
            if (is_m) begin
                if (f6 == 6'b100101) r = a * b;          // vmul
                else r = a * b + d;                      // vmacc: vd = vs1*vs2 + vd
            end else begin
                case (f6)
                    6'b000000: r = a + b;
                    6'b000010: r = a - b;
                    6'b000011: r = b - a;
                    6'b000100: r = (a < b) ? a : b;
                    6'b000101: r = ($signed(sa) < $signed(sb)) ? a : b;
                    6'b000110: r = (a > b) ? a : b;
                    6'b000111: r = ($signed(sa) > $signed(sb)) ? a : b;
                    6'b001001: r = a & b;
                    6'b001010: r = a | b;
                    6'b001011: r = a ^ b;
                    6'b100101: r = a << sh;
                    6'b101000: r = a >> sh;
                    6'b101001: r = $signed(sa) >>> sh;
                    6'b010111: r = b;
                    default: r = 32'd0;
                endcase
            end
            valu = r & vmask(sew);
        end
    endfunction
    function automatic logic vop_legal(input [5:0] f6, input [2:0] f3);
        begin
            vop_legal = 1'b0;
            case (f3)
                3'b000, 3'b100, 3'b011: begin // OPIVV, OPIVX, OPIVI
                    case (f6)
                        6'b000000, 6'b001001, 6'b001010, 6'b001011, 6'b100101,
                        6'b101000, 6'b101001, 6'b010111: vop_legal = 1'b1;
                        6'b000010, 6'b000100, 6'b000101, 6'b000110, 6'b000111: vop_legal = (f3 != 3'b011);
                        6'b000011: vop_legal = (f3 != 3'b000);
                        default: vop_legal = 1'b0;
                    endcase
                end
                3'b010: vop_legal = (f6 == 6'b000000) || (f6 == 6'b100101) || (f6 == 6'b101101) || (f6 == 6'b010000);
                3'b110: vop_legal = (f6 == 6'b100101) || (f6 == 6'b101101) || (f6 == 6'b010000);
                default: vop_legal = 1'b0;
            endcase
        end
    endfunction

    wire [31:0] vs_elem = velem(vregs[v_reg], vsew, v_idx);
    wire [31:0] vs_wdata = vs_elem << (8 * data_address[1:0]);
    wire [3:0]  vs_wstrb = vbytes_mask(vsew) << data_address[1:0];

    assign imem_valid = (state == ST_FETCH) && (pc[1:0] == 2'b00);
    assign imem_addr = pc;
    assign dmem_valid = (state == ST_LOAD) || (state == ST_LOAD2) || (state == ST_STORE) ||
                        (state == ST_STORE2) || (state == ST_VLOAD) || (state == ST_VSTORE);
    assign dmem_write = (state == ST_STORE) || (state == ST_STORE2) || (state == ST_VSTORE);
    assign dmem_addr  = fetching_second ? ((data_address & 32'hffff_fffc) + 32'd4)
                                        : (data_address & 32'hffff_fffc);
    assign dmem_wdata = (state == ST_VSTORE) ? vs_wdata : (state == ST_STORE2) ? store_data2 : store_data;
    assign dmem_wstrb = (state == ST_VSTORE) ? vs_wstrb : (state == ST_STORE2) ? store_strobes2 : store_strobes;
    assign halted = (state == ST_HALTED);

    task automatic raise_trap(input logic [3:0] cause, input logic [31:0] value);
        begin
            trap_valid <= 1'b1;
            trap_pc <= pc;
            trap_cause <= cause;
            trap_value <= value;
            mepc <= pc;
            mcause <= {28'd0, cause};
            mtval <= value;
            mstatus_mpie <= mstatus_mie;
            mstatus_mie <= 1'b0;
            if (HALT_ON_TRAP) begin
                state <= ST_HALTED;
            end else begin
                pc <= mtvec;
                state <= ST_FETCH;
            end
        end
    endtask

    task automatic retire(input logic [31:0] instruction, input logic [31:0] next_pc);
        begin
            retire_valid <= 1'b1;
            retire_pc <= pc;
            retire_insn <= instruction;
            retire_next_pc <= next_pc;
            instret_pending <= 1'b1;
        end
    endtask

    task automatic execute_csr(input [31:0] ins);
        logic [11:0] addr;
        logic [31:0] old, src, newv;
        logic [63:0] instret_eff;
        logic do_write, read_only, known;
        begin
            addr = ins[31:20];
            src = ins[14] ? {27'd0, ins[19:15]} : rs1_value;
            do_write = (ins[13:12] == 2'b01) || (ins[19:15] != 5'd0);
            read_only = (addr[11:10] == 2'b11);
            instret_eff = minstret + {63'd0, (instret_pending && !mcountinhibit[2])};
            known = 1'b1;
            old = 32'd0;
            case (addr)
                12'h300: old = {19'd0, 2'b11, 3'd0, mstatus_mpie, 3'd0, mstatus_mie, 3'd0};
                12'h301: old = 32'h4000_0100;
                12'h304: old = mie_reg;
                12'h305: old = mtvec;
                12'h320: old = mcountinhibit;
                12'h340: old = mscratch;
                12'h341: old = mepc;
                12'h342: old = mcause;
                12'h343: old = mtval;
                12'h344: old = 32'd0;
                12'hB00, 12'hC00: old = mcycle[31:0];
                12'hB80, 12'hC80: old = mcycle[63:32];
                12'hB02, 12'hC02: old = instret_eff[31:0];
                12'hB82, 12'hC82: old = instret_eff[63:32];
                12'hF11, 12'hF12, 12'hF13, 12'hF14: old = 32'd0;
                12'hC20: old = vl_reg;
                12'hC21: old = vtype_reg;
                12'hC22: old = 32'd16;
                default: known = 1'b0;
            endcase
            case (ins[13:12])
                2'b01: newv = src;
                2'b10: newv = old | src;
                default: newv = old & ~src;
            endcase
            if (!known || (do_write && read_only)) begin
                raise_trap(CAUSE_ILLEGAL, ins);
            end else begin
                if (rd != 5'd0) registers[rd] <= old;
                pc <= pc + 32'd4;
                retire(ins, pc + 32'd4);
                if (do_write) begin
                    case (addr)
                        12'h300: begin mstatus_mie <= newv[3]; mstatus_mpie <= newv[7]; end
                        12'h304: mie_reg <= newv & 32'h0000_0888;
                        12'h305: mtvec <= newv & 32'hffff_fffc;
                        12'h320: mcountinhibit <= newv & 32'h0000_0005;
                        12'h340: mscratch <= newv;
                        12'h341: mepc <= newv & 32'hffff_fffc;
                        12'h342: mcause <= newv;
                        12'h343: mtval <= newv;
                        12'hB00: mcycle <= {mcycle[63:32], newv};
                        12'hB80: mcycle <= {newv, mcycle[31:0]};
                        12'hB02: begin minstret <= {instret_eff[63:32], newv}; instret_pending <= 1'b0; end
                        12'hB82: begin minstret <= {newv, instret_eff[31:0]}; instret_pending <= 1'b0; end
                        default: ;
                    endcase
                end
            end
        end
    endtask

    task automatic execute_branch(input logic [31:0] instruction);
        logic take_branch;
        logic [31:0] target;
        begin
            case (funct3)
                3'b000: take_branch = rs1_value == rs2_value; // BEQ
                3'b001: take_branch = rs1_value != rs2_value; // BNE
                3'b100: take_branch = $signed(rs1_value) < $signed(rs2_value); // BLT
                3'b101: take_branch = $signed(rs1_value) >= $signed(rs2_value); // BGE
                3'b110: take_branch = rs1_value < rs2_value; // BLTU
                3'b111: take_branch = rs1_value >= rs2_value; // BGEU
                default: take_branch = 1'b0;
            endcase
            target = pc + imm_b;
            if (take_branch && target[1:0] != 2'b00) begin
                raise_trap(CAUSE_INSN_MISALIGNED, target);
            end else begin
                pc <= take_branch ? target : pc + 32'd4;
                retire(instruction, take_branch ? target : pc + 32'd4);
            end
        end
    endtask

    task automatic execute_imm(input logic [31:0] instruction);
        logic valid;
        logic [31:0] result;
        begin
            valid = 1'b1;
            result = 32'd0;
            case (funct3)
                3'b000: result = rs1_value + imm_i; // ADDI
                3'b010: result = ($signed(rs1_value) < $signed(imm_i)) ? 32'd1 : 32'd0; // SLTI
                3'b011: result = (rs1_value < imm_i) ? 32'd1 : 32'd0; // SLTIU
                3'b100: result = rs1_value ^ imm_i; // XORI
                3'b110: result = rs1_value | imm_i; // ORI
                3'b111: result = rs1_value & imm_i; // ANDI
                3'b001: begin // SLLI
                    valid = funct7 == 7'b0000000;
                    result = rs1_value << imem_rdata[24:20];
                end
                3'b101: begin // SRLI / SRAI
                    valid = (funct7 == 7'b0000000) || (funct7 == 7'b0100000);
                    if (funct7 == 7'b0100000)
                        result = $signed(rs1_value) >>> imem_rdata[24:20];
                    else result = rs1_value >> imem_rdata[24:20];
                end
                default: valid = 1'b0;
            endcase
            if (!valid) begin
                raise_trap(CAUSE_ILLEGAL, instruction);
            end else begin
                if (rd != 0) registers[rd] <= result;
                pc <= pc + 32'd4;
                retire(instruction, pc + 32'd4);
            end
        end
    endtask

    task automatic execute_reg(input logic [31:0] instruction);
        logic valid;
        logic [31:0] result;
        begin
            valid = 1'b1;
            result = 32'd0;
            case (funct3)
                3'b000: begin
                    if (funct7 == 7'b0000000) result = rs1_value + rs2_value; // ADD
                    else if (funct7 == 7'b0100000) result = rs1_value - rs2_value; // SUB
                    else valid = 1'b0;
                end
                3'b001: begin valid = funct7 == 0; result = rs1_value << rs2_value[4:0]; end // SLL
                3'b010: begin valid = funct7 == 0; result = $signed(rs1_value) < $signed(rs2_value); end // SLT
                3'b011: begin valid = funct7 == 0; result = rs1_value < rs2_value; end // SLTU
                3'b100: begin valid = funct7 == 0; result = rs1_value ^ rs2_value; end // XOR
                3'b101: begin
                    valid = (funct7 == 7'b0000000) || (funct7 == 7'b0100000);
                    if (funct7 == 7'b0100000) result = $signed(rs1_value) >>> rs2_value[4:0];
                    else result = rs1_value >> rs2_value[4:0];
                end
                3'b110: begin valid = funct7 == 0; result = rs1_value | rs2_value; end // OR
                3'b111: begin valid = funct7 == 0; result = rs1_value & rs2_value; end // AND
                default: valid = 1'b0;
            endcase
            if (!valid) begin
                raise_trap(CAUSE_ILLEGAL, instruction);
            end else begin
                if (rd != 0) registers[rd] <= result;
                pc <= pc + 32'd4;
                retire(instruction, pc + 32'd4);
            end
        end
    endtask

    task automatic execute_opv(input [31:0] ins);
        logic [5:0] f6;
        logic [2:0] f3;
        logic [1:0] sew;
        logic [4:0] vd, vs1, vs2;
        logic [127:0] vec, vsrc2, vsrc1;
        logic [31:0] a, b, d, avl, new_vl, new_vt, sum, vlmax, old_vlmax;
        logic vill_new, is_m, is_vsetivli;
        integer n, i;
        begin
            f6 = ins[31:26];
            f3 = ins[14:12];
            sew = vtype_reg[4:3];
            n = vl_reg;
            is_m = (f3 == 3'b010) || (f3 == 3'b110);
            if (f3 == 3'b111) begin
                is_vsetivli = ins[31] && ins[30];
                if (ins[31] && !ins[30] && ins[29:25] != 5'd0) begin
                    raise_trap(CAUSE_ILLEGAL, ins);
                end else begin
                    if (!ins[31]) new_vt = {21'd0, ins[30:20]};
                    else if (ins[30]) new_vt = {22'd0, ins[29:20]};
                    else new_vt = rs2_value;
                    vill_new = new_vt[31] || (new_vt[30:8] != 23'd0) || (new_vt[2:0] != 3'd0) || (new_vt[5:3] > 3'd2);
                    vlmax = vill_new ? 32'd0 : (32'd16 >> new_vt[4:3]);
                    old_vlmax = vtype_reg[31] ? 32'd0 : (32'd16 >> vtype_reg[4:3]);
                    if (is_vsetivli) avl = {27'd0, rs1};
                    else if (rs1 != 5'd0) avl = rs1_value;
                    else if (rd != 5'd0) avl = 32'hffff_ffff;
                    else avl = vl_reg;
                    if (vill_new || (rs1 == 5'd0 && rd == 5'd0 && !is_vsetivli && vlmax != old_vlmax)) begin
                        new_vl = 32'd0;
                        vtype_reg <= 32'h8000_0000;
                        vl_reg <= 32'd0;
                    end else begin
                        new_vl = (avl > vlmax) ? vlmax : avl;
                        vtype_reg <= {24'd0, new_vt[7:0]};
                        vl_reg <= new_vl;
                    end
                    if (rd != 5'd0) registers[rd] <= new_vl;
                    pc <= pc + 32'd4;
                    retire(ins, pc + 32'd4);
                end
            end else if (vtype_reg[31] || !ins[25] || !vop_legal(f6, f3)
                         || (f6 == 6'b010111 && !is_m && ins[24:20] != 5'd0)
                         || (f3 == 3'b010 && f6 == 6'b010000 && ins[19:15] != 5'd0)
                         || (f3 == 3'b110 && f6 == 6'b010000 && ins[24:20] != 5'd0)) begin
                raise_trap(CAUSE_ILLEGAL, ins);
            end else begin
                vd = ins[11:7];
                vs2 = ins[24:20];
                vs1 = ins[19:15];
                vsrc2 = vregs[vs2];
                vsrc1 = vregs[vs1];
                vec = vregs[vd];
                if (f3 == 3'b010 && f6 == 6'b010000) begin           // vmv.x.s
                    a = vsext(velem(vsrc2, sew, 0), sew);
                    if (rd != 5'd0) registers[rd] <= a;
                end else if (f3 == 3'b110 && f6 == 6'b010000) begin  // vmv.s.x
                    if (n > 0) vregs[vd] <= vput(vec, sew, 0, rs1_value);
                end else if (f3 == 3'b010 && f6 == 6'b000000) begin  // vredsum.vs
                    sum = velem(vsrc1, sew, 0);
                    for (i = 0; i < n; i = i + 1) sum = sum + velem(vsrc2, sew, i);
                    if (n > 0) vregs[vd] <= vput(vec, sew, 0, sum);
                end else begin
                    for (i = 0; i < n; i = i + 1) begin
                        a = velem(vsrc2, sew, i);
                        d = velem(vec, sew, i);
                        case (f3)
                            3'b000, 3'b010: b = velem(vsrc1, sew, i);
                            3'b100, 3'b110: b = rs1_value & vmask(sew);
                            default: b = {{27{ins[19]}}, ins[19:15]} & vmask(sew);
                        endcase
                        vec = vput(vec, sew, i, valu(f6, is_m, sew, a, b, d));
                    end
                    vregs[vd] <= vec;
                end
                pc <= pc + 32'd4;
                retire(ins, pc + 32'd4);
            end
        end
    endtask

    task automatic execute_vmem(input [31:0] ins);
        logic is_store, eew_ok;
        logic [1:0] eew;
        logic [31:0] addr, bytes;
        begin
            is_store = ins[5];
            case (ins[14:12])
                3'b000: begin eew = 2'd0; eew_ok = 1'b1; end
                3'b101: begin eew = 2'd1; eew_ok = 1'b1; end
                3'b110: begin eew = 2'd2; eew_ok = 1'b1; end
                default: begin eew = 2'd0; eew_ok = 1'b0; end
            endcase
            bytes = 32'd1 << eew;
            addr = rs1_value;
            if (!eew_ok || vtype_reg[31] || eew != vtype_reg[4:3] || ins[31:26] != 6'd0 ||
                !ins[25] || ins[24:20] != 5'd0) begin
                raise_trap(CAUSE_ILLEGAL, ins);
            end else if (vl_reg == 32'd0) begin
                pc <= pc + 32'd4;
                retire(ins, pc + 32'd4);
            end else if ((addr & (bytes - 32'd1)) != 32'd0) begin
                raise_trap(is_store ? CAUSE_STORE_MISALIGNED : CAUSE_LOAD_MISALIGNED, addr);
            end else begin
                data_address <= addr;
                v_idx <= 6'd0;
                v_reg <= ins[11:7];
                memory_instruction <= ins;
                state <= is_store ? ST_VSTORE : ST_VLOAD;
            end
        end
    endtask

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_FETCH;
            pc <= RESET_PC;
            data_address <= 32'd0;
            store_data <= 32'd0;
            store_strobes <= 4'd0;
            store_data2 <= 32'd0;
            store_strobes2 <= 4'd0;
            store_two <= 1'b0;
            load_funct3 <= 3'd0;
            load_rd <= 5'd0;
            load_shift <= 2'd0;
            load_two <= 1'b0;
            load_word0 <= 32'd0;
            memory_instruction <= 32'd0;
            trap_valid <= 1'b0;
            trap_pc <= 32'd0;
            trap_cause <= 4'd0;
            trap_value <= 32'd0;
            retire_valid <= 1'b0;
            retire_pc <= 32'd0;
            retire_insn <= 32'd0;
            retire_next_pc <= 32'd0;
            for (reset_index = 0; reset_index < 32; reset_index = reset_index + 1)
                registers[reset_index] <= 32'd0;
            for (reset_index = 0; reset_index < 32; reset_index = reset_index + 1)
                vregs[reset_index] <= 128'd0;
            vl_reg <= 32'd0;
            vtype_reg <= 32'h8000_0000;
            v_idx <= 6'd0;
            v_reg <= 5'd0;
            mstatus_mie <= 1'b0;
            mstatus_mpie <= 1'b0;
            mtvec <= 32'd0;
            mepc <= 32'd0;
            mcause <= 32'd0;
            mtval <= 32'd0;
            mscratch <= 32'd0;
            mie_reg <= 32'd0;
            mcountinhibit <= 32'd0;
            mcycle <= 64'd0;
            minstret <= 64'd0;
            instret_pending <= 1'b0;
        end else begin
            retire_valid <= 1'b0;
            instret_pending <= 1'b0;
            if (!HALT_ON_TRAP) trap_valid <= 1'b0;
            if (!mcountinhibit[0]) mcycle <= mcycle + 64'd1;
            if (instret_pending && !mcountinhibit[2]) minstret <= minstret + 64'd1;
            registers[0] <= 32'd0;
            case (state)
                ST_FETCH: begin
                    if (pc[1:0] != 2'b00) begin
                        raise_trap(CAUSE_INSN_MISALIGNED, pc);
                    end else if (imem_ready) begin
                        if (imem_fault) begin
                            raise_trap(CAUSE_INSN_ACCESS, pc);
                        end else begin
                            case (opcode)
                                7'b0110111: begin // LUI
                                    if (rd != 0) registers[rd] <= imm_u;
                                    pc <= pc + 32'd4;
                                    retire(imem_rdata, pc + 32'd4);
                                end
                                7'b0010111: begin // AUIPC
                                    if (rd != 0) registers[rd] <= pc + imm_u;
                                    pc <= pc + 32'd4;
                                    retire(imem_rdata, pc + 32'd4);
                                end
                                7'b1101111: begin // JAL
                                    if (jal_target[1:0] != 2'b00) begin
                                        raise_trap(CAUSE_INSN_MISALIGNED, jal_target);
                                    end else begin
                                        if (rd != 0) registers[rd] <= pc + 32'd4;
                                        pc <= jal_target;
                                        retire(imem_rdata, jal_target);
                                    end
                                end
                                7'b1100111: begin // JALR
                                    if (funct3 != 3'b000) begin
                                        raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                    end else if (jalr_target[1:0] != 2'b00) begin
                                        raise_trap(CAUSE_INSN_MISALIGNED, jalr_target);
                                    end else begin
                                        if (rd != 0) registers[rd] <= pc + 32'd4;
                                        pc <= jalr_target;
                                        retire(imem_rdata, jalr_target);
                                    end
                                end
                                7'b1100011: begin
                                    if (funct3 == 3'b010 || funct3 == 3'b011)
                                        raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                    else execute_branch(imem_rdata);
                                end
                                7'b0000011: begin // LB, LH, LW, LBU, LHU
                                    if (funct3 == 3'b000 || funct3 == 3'b001 || funct3 == 3'b010 ||
                                        funct3 == 3'b100 || funct3 == 3'b101) begin
                                        data_address <= load_effective_address;
                                        load_shift <= load_effective_address[1:0];
                                        load_two <= (funct3 != 3'b000) && (load_effective_address[1:0] != 2'b00);
                                        load_funct3 <= funct3;
                                        load_rd <= rd;
                                        memory_instruction <= imem_rdata;
                                        state <= ST_LOAD;
                                    end else raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                end
                                7'b0100011: begin // SB, SH, SW
                                    if (funct3 == 3'b000 || funct3 == 3'b001 || funct3 == 3'b010) begin
                                        data_address <= store_effective_address;
                                        store_two <= (store_shift + store_nbytes) > 3'd4;
                                        store_data <= rs2_value << (8 * store_shift);
                                        store_strobes <= ((((4'h1 << store_nbytes) - 4'h1) << store_shift) & 4'hf);
                                        store_data2 <= rs2_value >> (8 * (4'd4 - store_shift));
                                        store_strobes2 <= (((4'h1 << (store_shift + store_nbytes - 4'd4)) - 4'h1) & 4'hf);
                                        memory_instruction <= imem_rdata;
                                        state <= ST_STORE;
                                    end else raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                end
                                7'b0010011: execute_imm(imem_rdata);
                                7'b0110011: execute_reg(imem_rdata);
                                7'b0001111: begin // FENCE / FENCE.I: single in-order core, uncached fetch, so both are no-ops.
                                    // rd/rs1/fm are ignored per spec (reserved encodings behave as fence); ACT tests this.
                                    if (funct3 == 3'b000 || funct3 == 3'b001) begin
                                        pc <= pc + 32'd4;
                                        retire(imem_rdata, pc + 32'd4);
                                    end else raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                end
                                7'b1110011: begin // SYSTEM
                                    if (imem_rdata == 32'h0000_0073)
                                        raise_trap(CAUSE_ENV_CALL, 32'd0);
                                    else if (imem_rdata == 32'h0010_0073)
                                        raise_trap(CAUSE_BREAKPOINT, pc);
                                    else if (imem_rdata == 32'h3020_0073) begin // MRET
                                        mstatus_mie <= mstatus_mpie;
                                        mstatus_mpie <= 1'b1;
                                        pc <= mepc;
                                        retire(imem_rdata, mepc);
                                    end else if (imem_rdata == 32'h1050_0073) begin // WFI: no interrupt sources, retire as nop
                                        pc <= pc + 32'd4;
                                        retire(imem_rdata, pc + 32'd4);
                                    end else if (funct3 != 3'b000 && funct3 != 3'b100) execute_csr(imem_rdata);
                                    else raise_trap(CAUSE_ILLEGAL, imem_rdata);
                                end
                                7'b1010111: execute_opv(imem_rdata);
                                7'b0000111, 7'b0100111: execute_vmem(imem_rdata);
                                default: raise_trap(CAUSE_ILLEGAL, imem_rdata);
                            endcase
                        end
                    end
                end
                ST_LOAD: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_LOAD_ACCESS, data_address);
                        end else if (load_two) begin
                            load_word0 <= dmem_rdata;
                            state <= ST_LOAD2;
                        end else begin
                            if (load_rd != 0) registers[load_rd] <= load_result;
                            pc <= pc + 32'd4;
                            state <= ST_FETCH;
                            retire(memory_instruction, pc + 32'd4);
                        end
                    end
                end
                ST_LOAD2: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_LOAD_ACCESS, data_address);
                        end else begin
                            if (load_rd != 0) registers[load_rd] <= load_result;
                            pc <= pc + 32'd4;
                            state <= ST_FETCH;
                            retire(memory_instruction, pc + 32'd4);
                        end
                    end
                end
                ST_STORE: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_STORE_ACCESS, data_address);
                        end else if (store_two) begin
                            state <= ST_STORE2;
                        end else begin
                            pc <= pc + 32'd4;
                            state <= ST_FETCH;
                            retire(memory_instruction, pc + 32'd4);
                        end
                    end
                end
                ST_STORE2: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_STORE_ACCESS, data_address);
                        end else begin
                            pc <= pc + 32'd4;
                            state <= ST_FETCH;
                            retire(memory_instruction, pc + 32'd4);
                        end
                    end
                end
                ST_VLOAD: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_LOAD_ACCESS, data_address);
                        end else begin
                            vregs[v_reg] <= vput(vregs[v_reg], vsew, v_idx, shifted_load_word);
                            if (v_idx + 1 >= vl_reg) begin
                                pc <= pc + 32'd4;
                                state <= ST_FETCH;
                                retire(memory_instruction, pc + 32'd4);
                            end else begin
                                v_idx <= v_idx + 6'd1;
                                data_address <= data_address + (32'd1 << vsew);
                            end
                        end
                    end
                end
                ST_VSTORE: begin
                    if (dmem_ready) begin
                        if (dmem_fault) begin
                            raise_trap(CAUSE_STORE_ACCESS, data_address);
                        end else if (v_idx + 1 >= vl_reg) begin
                            pc <= pc + 32'd4;
                            state <= ST_FETCH;
                            retire(memory_instruction, pc + 32'd4);
                        end else begin
                            v_idx <= v_idx + 6'd1;
                            data_address <= data_address + (32'd1 << vsew);
                        end
                    end
                end
                ST_HALTED: begin
                    // Trap metadata remains sticky until reset.
                end
                default: raise_trap(CAUSE_ILLEGAL, 32'd0);
            endcase
        end
    end
endmodule
